#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.preprocessing import LabelEncoder

from config import (
    DEFAULT_DATA,
    DEFAULT_MODEL_DIR,
    MODEL_FILENAME,
    PRECIP_THRESHOLD_MM,
)
from train_validate_test_split import Train_validate_test_split

BUNDLE_VERSION = "1.0"

# WMO 4677 present-weather codes, bucketed into the categories we forecast.
WMO_CATEGORIES: dict[str, tuple[int, ...]] = {
    "Clear": (0, 1),
    "Cloudy": (2, 3),
    "Fog": (45, 48),
    "Drizzle": (51, 53, 55, 56, 57),
    "Rain": (61, 63, 65, 66, 67, 80, 81, 82),
    "Snow": (71, 73, 75, 77, 85, 86),
    "Thunderstorm": (95, 96, 99),
}
WMO_TO_CATEGORY = {
    code: category for category, codes in WMO_CATEGORIES.items() for code in codes
}
UNKNOWN_CATEGORY = "Unknown"

# Feature-engineering knobs. The widest window (30) sets the warm-up period the first 29 days of each city are dropped for want of a full history.
LAG_DAYS = (1, 2, 3, 7)
ROLL_WINDOWS = (3, 7, 14, 30)
LAGGED_COLUMNS = ("temp_mean_c", "temp_max_c", "temp_min_c", "precip_mm", "temp_range_c")

TARGET_TEMP = "y_temp_mean_c"
TARGET_PRECIP = "y_is_wet"
TARGET_CONDITION = "y_condition_code"
TARGET_COLUMNS = (TARGET_TEMP, TARGET_PRECIP, TARGET_CONDITION)

# Bookkeeping columns carried alongside the features but never fed to a model.
META_COLUMNS = (
    "city",
    "date",
    "target_date",
    "condition",
    "y_condition",
    "y_precip_mm",
)


def convert_names(name: str) -> str:
    folded = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return "".join(ch if ch.isalnum() else "_" for ch in folded.lower()).strip("_")

def categorize_wmo(code) -> str:
    if pd.isna(code):
        return UNKNOWN_CATEGORY
    return WMO_TO_CATEGORY.get(int(code), UNKNOWN_CATEGORY)

# Hard coded to CSV while we wait for DB to be operational. The CSV is built by fetch_weather.py and consumed by train.py and predict_weather.py.
def load_raw(data_path: str | Path) -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python fetch_weather.py` first to build it"
        )

    df = pd.read_csv(path, encoding="utf-8")
    missing = {"city", "date", "temp_mean_c", "temp_max_c", "temp_min_c",
               "precip_mm", "wmo_code"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s): {', '.join(sorted(missing))}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["city", "date"]).drop_duplicates(["city", "date"])
    return df.reset_index(drop=True)


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["condition"] = df["wmo_code"].map(categorize_wmo)
    df["temp_range_c"] = df["temp_max_c"] - df["temp_min_c"]
    df["is_wet"] = (df["precip_mm"] > PRECIP_THRESHOLD_MM).astype(float)

    grouped = df.groupby("city", sort=False)

    # the history of yesterday, the day before, a week ago#
    for column in LAGGED_COLUMNS:
        for lag in LAG_DAYS:
            df[f"{column}_lag{lag}"] = grouped[column].shift(lag)

    # trailing windows (inclusive of today) #
    def roll(column: str, window: int, how: str) -> pd.Series:
        return grouped[column].transform(
            lambda s: getattr(s.rolling(window, min_periods=window), how)()
        )

    for window in ROLL_WINDOWS:
        df[f"temp_mean_c_roll{window}_mean"] = roll("temp_mean_c", window, "mean")
        df[f"precip_mm_roll{window}_mean"] = roll("precip_mm", window, "mean")
    df["temp_mean_c_roll7_std"] = roll("temp_mean_c", 7, "std")
    df["temp_mean_c_roll30_std"] = roll("temp_mean_c", 30, "std")
    df["temp_range_c_roll7_mean"] = roll("temp_range_c", 7, "mean")
    df["precip_mm_roll3_sum"] = roll("precip_mm", 3, "sum")
    df["precip_mm_roll7_sum"] = roll("precip_mm", 7, "sum")
    df["wet_days_roll7"] = roll("is_wet", 7, "sum")
    df["wet_days_roll30"] = roll("is_wet", 30, "sum")

    # where today sits against its own recent history#
    df["temp_delta_1d"] = df["temp_mean_c"] - df["temp_mean_c_lag1"]
    df["temp_delta_3d"] = df["temp_mean_c"] - df["temp_mean_c_lag3"]
    df["temp_anomaly_7d"] = df["temp_mean_c"] - df["temp_mean_c_roll7_mean"]
    df["temp_anomaly_30d"] = df["temp_mean_c"] - df["temp_mean_c_roll30_mean"]
    df["precip_delta_1d"] = df["precip_mm"] - df["precip_mm_lag1"]

    # today's sky, one-hot (trees shouldn't read order into categories) #
    for category in WMO_CATEGORIES:
        df[f"cond_today_{convert_names(category)}"] = (df["condition"] == category).astype(float)

    #the day being forecast, and where #
    df["target_date"] = df["date"] + pd.Timedelta(days=1)
    target_doy = df["target_date"].dt.dayofyear
    df["target_doy_sin"] = np.sin(2 * np.pi * target_doy / 365.25)
    df["target_doy_cos"] = np.cos(2 * np.pi * target_doy / 365.25)
    df["target_month"] = df["target_date"].dt.month.astype(float)
    # Southern-hemisphere cities run half a year out of phase with northern ones; signing the seasonal wave by hemisphere lets one split serve both.
    df["hemisphere_doy_sin"] = df["target_doy_sin"] * np.sign(df["latitude"])

    for city in sorted(raw["city"].unique()):
        df[f"city_{convert_names(city)}"] = (df["city"] == city).astype(float)

    next_date = grouped["date"].shift(-1)
    is_consecutive = (next_date - df["date"]) == pd.Timedelta(days=1)

    df[TARGET_TEMP] = grouped["temp_mean_c"].shift(-1).where(is_consecutive)
    df["y_precip_mm"] = grouped["precip_mm"].shift(-1).where(is_consecutive)
    df[TARGET_PRECIP] = (df["y_precip_mm"] > PRECIP_THRESHOLD_MM).astype(float)
    df.loc[df["y_precip_mm"].isna(), TARGET_PRECIP] = np.nan
    df["y_condition"] = grouped["condition"].shift(-1).where(is_consecutive)

    # Warm-up rows lack a full 30-day window and can never be scored or served.
    df = df.dropna(subset=[f"temp_mean_c_roll{max(ROLL_WINDOWS)}_mean"])
    return df.reset_index(drop=True)


def feature_names(features: pd.DataFrame) -> list[str]:

    excluded = set(META_COLUMNS) | set(TARGET_COLUMNS)
    return [
        column
        for column in features.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(features[column])
    ]


SPLIT_STRATEGY = "random 64/16/20 (Train_validate_test_split)"


COMMON_PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_lambda=1.0,
    early_stopping_rounds=50,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)


def train_temperature_model(X_train, y_train, X_validate, y_validate) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(objective="reg:squarederror", eval_metric="rmse",
                             **COMMON_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_validate, y_validate)], verbose=False)
    return model


def train_precip_model(X_train, y_train, X_validate, y_validate) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                              **COMMON_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_validate, y_validate)], verbose=False)
    return model


def train_condition_model(X_train, y_train, X_validate, y_validate,
                          n_classes: int) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(objective="multi:softprob", eval_metric="mlogloss",
                              num_class=n_classes, **COMMON_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_validate, y_validate)], verbose=False)
    return model


def evaluate_temperature(model, X, y_true, persistence: pd.Series) -> dict:
    predicted = model.predict(X)
    return {
        "mae_c": float(mean_absolute_error(y_true, predicted)),
        "rmse_c": float(root_mean_squared_error(y_true, predicted)),
        "r2": float(r2_score(y_true, predicted)),
        "baseline_persistence_mae_c": float(mean_absolute_error(y_true, persistence)),
    }


def evaluate_precip(model, X, y_true) -> dict:
    probability = model.predict_proba(X)[:, 1]
    predicted = (probability >= 0.5).astype(int)
    base_rate = float(np.mean(y_true))
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, probability)),
        "base_rate": base_rate,
        "baseline_majority_accuracy": max(base_rate, 1 - base_rate),
    }


def evaluate_condition(model, X, y_true, classes: list[str]) -> dict:
    probability = model.predict_proba(X)
    predicted = probability.argmax(axis=1)
    counts = np.bincount(np.asarray(y_true, dtype=int), minlength=len(classes))
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro",
                                   labels=range(len(classes)), zero_division=0)),
        "log_loss": float(log_loss(y_true, probability, labels=range(len(classes)))),
        "baseline_majority_accuracy": float(counts.max() / counts.sum()),
    }


def top_features(model, names: list[str], k: int = 12) -> list[tuple[str, float]]:
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:k]
    return [(names[i], float(importances[i])) for i in order]


def report(metrics: dict, importances: dict) -> None:
    temp = metrics["temperature"]
    precip = metrics["precipitation"]
    condition = metrics["condition"]

    print()
    print("=" * 68)
    print("Next-day temperature (deg C)".ljust(30) + f"[{temp['best_iteration'] + 1} trees]")
    print("-" * 68)
    print(f"{'':12}{'MAE':>10}{'RMSE':>10}{'R2':>10}{'persistence MAE':>20}")
    for split in ("validate", "test"):
        s = temp[split]
        print(f"{split:12}{s['mae_c']:>10.3f}{s['rmse_c']:>10.3f}{s['r2']:>10.3f}"
              f"{s['baseline_persistence_mae_c']:>20.3f}")

    print()
    print("Next-day precipitation (P[> 0.1 mm])".ljust(30)
          + f"[{precip['best_iteration'] + 1} trees]")
    print("-" * 68)
    print(f"{'':12}{'accuracy':>10}{'ROC AUC':>10}{'log loss':>10}{'Brier':>10}{'majority':>12}")
    for split in ("validate", "test"):
        s = precip[split]
        print(f"{split:12}{s['accuracy']:>10.3f}{s['roc_auc']:>10.3f}{s['log_loss']:>10.3f}"
              f"{s['brier']:>10.3f}{s['baseline_majority_accuracy']:>12.3f}")

    print()
    print("Next-day condition category".ljust(30)
          + f"[{condition['best_iteration'] + 1} trees]")
    print("-" * 68)
    print(f"{'':12}{'accuracy':>10}{'macro F1':>10}{'log loss':>10}{'majority':>12}")
    for split in ("validate", "test"):
        s = condition[split]
        print(f"{split:12}{s['accuracy']:>10.3f}{s['macro_f1']:>10.3f}{s['log_loss']:>10.3f}"
              f"{s['baseline_majority_accuracy']:>12.3f}")

    print()
    print("Top features")
    print("-" * 68)
    for target, ranked in importances.items():
        top = ", ".join(f"{name} ({score:.3f})" for name, score in ranked[:6])
        print(f"  {target:14} {top}")
    print("=" * 68)


# --------------------------------------------------------------------------- #
# Training entry point
# --------------------------------------------------------------------------- #
def save_bundle(bundle: dict, model_dir: str | Path) -> Path:
    directory = Path(model_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MODEL_FILENAME
    joblib.dump(bundle, path)
    return path


def train(data_path: str = DEFAULT_DATA, model_dir: str = DEFAULT_MODEL_DIR,
          quiet: bool = False) -> dict:
    """Fit all three next-day models and write the bundle to `model_dir`."""

    def say(message: str = "") -> None:
        if not quiet:
            print(message)

    raw = load_raw(data_path)
    say(f"Loaded {len(raw):,} daily observations for {raw['city'].nunique()} cities "
        f"({raw['date'].min().date()} -> {raw['date'].max().date()})")

    features = build_features(raw)
    labelled = features.dropna(subset=[TARGET_TEMP, TARGET_PRECIP, "y_condition"]).copy()

    encoder = LabelEncoder().fit(labelled["y_condition"])
    labelled[TARGET_CONDITION] = encoder.transform(labelled["y_condition"])
    condition_classes = list(encoder.classes_)

    names = feature_names(labelled)
    X = labelled[names]
    y = labelled[list(TARGET_COLUMNS)]

    say(f"Built {len(X):,} supervised rows x {len(names)} features")
    say(f"Condition classes: {', '.join(condition_classes)}")
    say(f"Wet-day rate: {labelled[TARGET_PRECIP].mean():.1%}")
    say(f"Split: {SPLIT_STRATEGY} -- neighbouring days can land on either side, "
        "so these scores flatter the model somewhat")
    say()

    X_train, X_validate, X_test, y_train, y_validate, y_test = Train_validate_test_split(X, y)
    say(f"Split -> train {len(X_train):,} | validate {len(X_validate):,} | test {len(X_test):,}")
    say()

    persistence = {
        "validate": X_validate["temp_mean_c"],
        "test": X_test["temp_mean_c"],
    }

    say("Training temperature regressor ...")
    temp_model = train_temperature_model(
        X_train, y_train[TARGET_TEMP], X_validate, y_validate[TARGET_TEMP])

    say("Training precipitation classifier ...")
    precip_model = train_precip_model(
        X_train, y_train[TARGET_PRECIP], X_validate, y_validate[TARGET_PRECIP])

    say("Training condition classifier ...")
    condition_model = train_condition_model(
        X_train, y_train[TARGET_CONDITION], X_validate, y_validate[TARGET_CONDITION],
        n_classes=len(condition_classes))

    metrics = {
        "temperature": {
            "best_iteration": int(temp_model.best_iteration),
            "validate": evaluate_temperature(temp_model, X_validate,
                                             y_validate[TARGET_TEMP], persistence["validate"]),
            "test": evaluate_temperature(temp_model, X_test,
                                         y_test[TARGET_TEMP], persistence["test"]),
        },
        "precipitation": {
            "best_iteration": int(precip_model.best_iteration),
            "validate": evaluate_precip(precip_model, X_validate, y_validate[TARGET_PRECIP]),
            "test": evaluate_precip(precip_model, X_test, y_test[TARGET_PRECIP]),
        },
        "condition": {
            "best_iteration": int(condition_model.best_iteration),
            "validate": evaluate_condition(condition_model, X_validate,
                                           y_validate[TARGET_CONDITION], condition_classes),
            "test": evaluate_condition(condition_model, X_test,
                                       y_test[TARGET_CONDITION], condition_classes),
        },
    }

    if not quiet:
        report(metrics, {
            "temperature": top_features(temp_model, names),
            "precipitation": top_features(precip_model, names),
            "condition": top_features(condition_model, names),
        })

    cities = {
        city: {"latitude": float(row["latitude"]), "longitude": float(row["longitude"])}
        for city, row in raw.groupby("city")[["latitude", "longitude"]].first().iterrows()
    }

    bundle = {
        "bundle_version": BUNDLE_VERSION,
        "temp_model": temp_model,
        "precip_model": precip_model,
        "condition_model": condition_model,
        "feature_names": names,
        "condition_classes": condition_classes,
        "cities": cities,
        "data_path": str(data_path),
        "split": SPLIT_STRATEGY,
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    saved_to = save_bundle(bundle, model_dir)
    say(f"Saved models to {saved_to}")
    say(f'Forecast with: python predict_weather.py --city "New York" '
        f'--date {raw["date"].max().date()}')
    return bundle

# CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help=f"observation CSV (default: {DEFAULT_DATA})")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                        help=f"where to save the model bundle (default: {DEFAULT_MODEL_DIR})")
    parser.add_argument("--quiet", action="store_true", help="suppress the metrics report")
    return parser

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        train(data_path=args.data, model_dir=args.model_dir, quiet=args.quiet)
    except (FileNotFoundError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
