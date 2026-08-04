#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

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
    CITY_COORDINATES,
    DATA_SOURCES,
    DEFAULT_DATA,
    DEFAULT_DATA_SOURCE,
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

REQUIRED_COLUMNS = ("city", "date", "temp_mean_c", "temp_max_c", "temp_min_c",
                    "precip_mm", "wmo_code")


# One row per city-day, whatever the origin: the CSV from fetch_weather.py or the SQL rollup in db.py. Both are validated and ordered here so build_features() only ever sees one shape.
def normalize_raw(df: pd.DataFrame, origin: str) -> pd.DataFrame:
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{origin} is missing column(s): {', '.join(sorted(missing))}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["city", "date"]).drop_duplicates(["city", "date"])
    return df.reset_index(drop=True)


def load_raw(data_path: str | Path) -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python fetch_weather.py` first to build it, "
            "or read the database instead with --source db"
        )
    return normalize_raw(pd.read_csv(path, encoding="utf-8"), str(path))


def coordinates(path: str | Path = CITY_COORDINATES) -> dict[str, tuple[float, float]]:
    """City coordinates keyed by folded name, from the cache fetch_weather.py writes."""
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(
            f"{file} not found -- the database stores city names but no coordinates, "
            "so run `python fetch_weather.py` once to build the geocoding cache"
        )
    cache = json.loads(file.read_text(encoding="utf-8"))
    return {
        convert_names(entry["name"]): (float(entry["latitude"]), float(entry["longitude"]))
        for entry in cache.values()
        if entry.get("latitude") is not None and entry.get("longitude") is not None
    }


def attach_coordinates(frame: pd.DataFrame,
                       path: str | Path = CITY_COORDINATES) -> pd.DataFrame:
    # latitude is load-bearing -- it signs the seasonal wave by hemisphere -- so an unmatched city is an error rather than a silently missing feature. Names are folded so "Sao Paulo" in the database still matches "São Paulo" in the cache.
    known = coordinates(path)
    folded = frame["city"].map(convert_names)
    missing = sorted(set(frame.loc[~folded.isin(known), "city"]))
    if missing:
        raise ValueError(
            f"no coordinates for {', '.join(repr(city) for city in missing)} in {path} "
            "-- refresh it with `python fetch_weather.py --refresh`"
        )

    frame = frame.copy()
    # Inserted where the CSV keeps them, so both sources hand build_features() the same column order.
    frame.insert(1, "latitude", folded.map(lambda name: known[name][0]))
    frame.insert(2, "longitude", folded.map(lambda name: known[name][1]))
    return frame


def load_db() -> pd.DataFrame:
    # Imported here so the CSV path never needs psycopg2 installed.
    from db import DbConfig, fetch_observations

    config = DbConfig.from_env()
    frame = fetch_observations(config)

    # Everything downstream keys a city by name, so two cities sharing one would be silently deduplicated into a single half-complete series.
    repeated = frame.groupby(["city", "date"]).size()
    clashing = sorted({city for city, _ in repeated[repeated > 1].index})
    if clashing:
        raise ValueError(
            f"{config.describe()} has more than one city row named "
            f"{', '.join(repr(city) for city in clashing)} -- give them distinct "
            "cities.city_name values before training"
        )
    return normalize_raw(attach_coordinates(frame), config.describe())


def load_observations(source: str | None = None,
                      data_path: str | Path = DEFAULT_DATA) -> pd.DataFrame:
    source = (source or DEFAULT_DATA_SOURCE).strip().lower()
    if source not in DATA_SOURCES:
        raise ValueError(
            f"unknown data source {source!r}; expected one of: {', '.join(DATA_SOURCES)}"
        )
    return load_db() if source == "db" else load_raw(data_path)


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["condition"] = df["wmo_code"].map(categorize_wmo)
    df["temp_range_c"] = df["temp_max_c"] - df["temp_min_c"]
    df["is_wet"] = (df["precip_mm"] > PRECIP_THRESHOLD_MM).astype(float)

    grouped = df.groupby("city", sort=False)

    # the history of yesterday, the day before, a week ago
    for column in LAGGED_COLUMNS:
        for lag in LAG_DAYS:
            df[f"{column}_lag{lag}"] = grouped[column].shift(lag)

    # trailing windows (inclusive of today) 
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

    # where today sits against its own recent history
    df["temp_delta_1d"] = df["temp_mean_c"] - df["temp_mean_c_lag1"]
    df["temp_delta_3d"] = df["temp_mean_c"] - df["temp_mean_c_lag3"]
    df["temp_anomaly_7d"] = df["temp_mean_c"] - df["temp_mean_c_roll7_mean"]
    df["temp_anomaly_30d"] = df["temp_mean_c"] - df["temp_mean_c_roll30_mean"]
    df["precip_delta_1d"] = df["precip_mm"] - df["precip_mm_lag1"]

    # today's sky, one-hot (trees shouldn't read order into categories) 
    for category in WMO_CATEGORIES:
        df[f"cond_today_{convert_names(category)}"] = (df["condition"] == category).astype(float)

    #the day being forecast, and where 
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


# Held constant across the search.
FIXED_PARAMS = dict(
    early_stopping_rounds=50,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)

# The candidates. Each spans a different trade-off between how much structure a tree may fit (max_depth, min_child_weight) and how hard the fit is damped (learning_rate, subsample, colsample_bytree, reg_lambda), rather than being a dense grid over one axis.
PARAM_GRID = (
    {"name": "shallow",      "n_estimators": 400,  "max_depth": 3,  "learning_rate": 0.10,
     "min_child_weight": 1,  "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 1.0},
    {"name": "baseline",     "n_estimators": 800,  "max_depth": 5,  "learning_rate": 0.05,
     "min_child_weight": 3,  "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
    {"name": "slow-steady",  "n_estimators": 2000, "max_depth": 4,  "learning_rate": 0.02,
     "min_child_weight": 2,  "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0},
    {"name": "wide-sparse",  "n_estimators": 1500, "max_depth": 6,  "learning_rate": 0.03,
     "min_child_weight": 5,  "subsample": 0.6, "colsample_bytree": 0.5, "reg_lambda": 2.0},
    {"name": "heavy-reg",    "n_estimators": 800,  "max_depth": 6,  "learning_rate": 0.05,
     "min_child_weight": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 20.0},
    {"name": "deep",         "n_estimators": 600,  "max_depth": 8,  "learning_rate": 0.05,
     "min_child_weight": 3,  "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
    {"name": "deep-damped",  "n_estimators": 1500, "max_depth": 8,  "learning_rate": 0.02,
     "min_child_weight": 10, "subsample": 0.7, "colsample_bytree": 0.6, "reg_lambda": 5.0},
    {"name": "aggressive",   "n_estimators": 300,  "max_depth": 10, "learning_rate": 0.10,
     "min_child_weight": 1,  "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 0.5},
    {"name": "short-budget", "n_estimators": 60,   "max_depth": 5,  "learning_rate": 0.10,
     "min_child_weight": 3,  "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
    {"name": "long-slow",    "n_estimators": 2500, "max_depth": 4,  "learning_rate": 0.01,
     "min_child_weight": 2,  "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0},
)

SELECTION_METRIC = {
    "temperature": "mae_c",
    "precipitation": "log_loss",
    "condition": "log_loss",
}


def make_temperature_model(params: dict) -> xgb.XGBRegressor:
    # Early stopping watches MAE so it optimises the same quantity the test metric reports. The model itself is trained on squared error, which is the only supported loss for regression trees in XGBoost.
    return xgb.XGBRegressor(objective="reg:squarederror", eval_metric="mae",
                            **FIXED_PARAMS, **params)


def make_precip_model(params: dict) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                             **FIXED_PARAMS, **params)


def make_condition_model(n_classes: int):
    def factory(params: dict) -> xgb.XGBClassifier:
        return xgb.XGBClassifier(objective="multi:softprob", eval_metric="mlogloss",
                                 num_class=n_classes, **FIXED_PARAMS, **params)
    return factory


def learning_curve(model, points: int = 8) -> list[tuple[int, float]]:
    history = model.evals_result_.get("validation_0", {})
    if not history:
        return []
    values = history[list(history)[-1]]
    step = max(1, len(values) // points)
    rounds = sorted({*range(0, len(values), step), int(model.best_iteration)})
    return [(i + 1, float(values[i])) for i in rounds]


def search(factory, evaluate, metric: str, X_train, y_train, X_validate, y_validate,
           grid=None, say=print) -> list[dict]:
    grid = PARAM_GRID if grid is None else grid

    trials = []
    for candidate in grid:
        params = {key: value for key, value in candidate.items() if key != "name"}
        started = perf_counter()
        model = factory(params)
        model.fit(X_train, y_train, eval_set=[(X_validate, y_validate)], verbose=False)
        scores = evaluate(model, X_validate, y_validate)
        trees = int(model.best_iteration) + 1
        # The budget bound if training ran out of trees before early stopping fired -- the candidate may still have been improving.
        capped = trees >= params["n_estimators"]
        trials.append({
            "name": candidate["name"],
            "params": params,
            "model": model,
            "best_iteration": int(model.best_iteration),
            "capped": capped,
            "validate": scores,
            "score": float(scores[metric]),
            "seconds": round(perf_counter() - started, 1),
            "curve": learning_curve(model),
        })
        say(f"    {candidate['name']:<13} {metric} {scores[metric]:.4f}"
            f"  ({trees} trees{' — budget bound' if capped else ''},"
            f" {trials[-1]['seconds']}s)")

    trials.sort(key=lambda trial: trial["score"])
    return trials


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


SEARCH_COLUMNS = {
    "temperature": (("mae_c", "MAE"), ("rmse_c", "RMSE"), ("r2", "R2")),
    "precipitation": (("log_loss", "log loss"), ("roc_auc", "ROC AUC"),
                      ("accuracy", "accuracy"), ("brier", "Brier")),
    "condition": (("log_loss", "log loss"), ("accuracy", "accuracy"),
                  ("macro_f1", "macro F1")),
}


def report_search(metrics: dict) -> None:
    for target, entry in metrics.items():
        trials = entry["trials"]
        columns = SEARCH_COLUMNS[target]
        print()
        print("=" * 84)
        print(f"{target} -- {len(trials)} candidates, "
              f"selected on validation {entry['selection_metric']}")
        print("-" * 84)
        header = (f"{'config':<14}{'depth':>6}{'lr':>7}{'mcw':>5}{'sub':>6}{'col':>6}"
                  f"{'lambda':>8}{'budget':>8}{'trees':>7}")
        print(header + "".join(f"{title:>10}" for _, title in columns))
        for trial in trials:
            params = trial["params"]
            trees = f"{trial['best_iteration'] + 1}{'*' if trial['capped'] else ''}"
            row = (f"{trial['name']:<14}{params['max_depth']:>6}"
                   f"{params['learning_rate']:>7.2f}{params['min_child_weight']:>5}"
                   f"{params['subsample']:>6.1f}{params['colsample_bytree']:>6.1f}"
                   f"{params['reg_lambda']:>8.1f}{params['n_estimators']:>8}{trees:>7}")
            row += "".join(f"{trial['validate'][key]:>10.3f}" for key, _ in columns)
            print(row + ("   <- best" if trial is trials[0] else ""))
        if any(trial["capped"] for trial in trials):
            print("  * budget bound before early stopping fired")

        curve = entry["learning_curve"]
        if curve:
            print(f"  winner's validation score by round: "
                  + "  ".join(f"{round_no}:{value:.3f}" for round_no, value in curve))


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


def save_bundle(bundle: dict, model_dir: str | Path) -> Path:
    directory = Path(model_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MODEL_FILENAME
    joblib.dump(bundle, path)
    return path


def train(data_path: str = DEFAULT_DATA, model_dir: str = DEFAULT_MODEL_DIR,
          quiet: bool = False, source: str | None = None) -> dict:

    def say(message: str = "") -> None:
        if not quiet:
            print(message)

    source = (source or DEFAULT_DATA_SOURCE).strip().lower()
    raw = load_observations(source, data_path)
    say(f"Source: {source} ({'database' if source == 'db' else data_path})")
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

    say(f"Trying {len(PARAM_GRID)} parameter sets per target, "
        f"selecting on validation {' / '.join(sorted(set(SELECTION_METRIC.values())))}")
    say()

    # (target, label, column, model factory, validation scorer, test scorer)
    plan = (
        ("temperature", "next-day temperature", TARGET_TEMP, make_temperature_model,
         lambda m, X, y: evaluate_temperature(m, X, y, persistence["validate"]),
         lambda m, X, y: evaluate_temperature(m, X, y, persistence["test"])),
        ("precipitation", "next-day precipitation", TARGET_PRECIP, make_precip_model,
         evaluate_precip, evaluate_precip),
        ("condition", "next-day condition", TARGET_CONDITION,
         make_condition_model(len(condition_classes)),
         lambda m, X, y: evaluate_condition(m, X, y, condition_classes),
         lambda m, X, y: evaluate_condition(m, X, y, condition_classes)),
    )

    models: dict[str, object] = {}
    metrics: dict[str, dict] = {}
    for target, label, column, factory, score_validate, score_test in plan:
        say(f"Searching {label} ...")
        metric = SELECTION_METRIC[target]
        trials = search(factory, score_validate, metric,
                        X_train, y_train[column], X_validate, y_validate[column],
                        say=say)
        best = trials[0]
        models[target] = best["model"]
        say(f"  best: {best['name']} ({metric} {best['score']:.4f})")
        say()

        metrics[target] = {
            "config": best["name"],
            "params": best["params"],
            "best_iteration": best["best_iteration"],
            "validate": best["validate"],
            "test": score_test(best["model"], X_test, y_test[column]),
            "selection_metric": metric,
            "learning_curve": best["curve"],
            # Every candidate, model objects dropped this keeps the bundle small while leaving the search auditable after the fact.
            "trials": [
                {key: trial[key] for key in
                 ("name", "params", "best_iteration", "capped", "score", "seconds",
                  "validate")}
                for trial in trials
            ],
        }

    temp_model = models["temperature"]
    precip_model = models["precipitation"]
    condition_model = models["condition"]

    if not quiet:
        report_search(metrics)
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
        # data_path stays the CSV fallback even for a database-trained bundle; data_source is what predict_weather.py reads by default.
        "data_path": str(data_path),
        "data_source": source,
        "split": SPLIT_STRATEGY,
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    saved_to = save_bundle(bundle, model_dir)
    say("Saved best model per target to " + str(saved_to) + ": "
        + ", ".join(f"{target}={entry['config']}" for target, entry in metrics.items()))
    say(f'Forecast with: python predict_weather.py --city "New York" '
        f'--date {raw["date"].max().date()}')
    return bundle

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=DATA_SOURCES, default=DEFAULT_DATA_SOURCE,
                        help="where observations come from; the database is configured "
                             f"in .env (default: {DEFAULT_DATA_SOURCE}, from DATA_SOURCE)")
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help=f"observation CSV, used by --source csv (default: {DEFAULT_DATA})")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                        help=f"where to save the model bundle (default: {DEFAULT_MODEL_DIR})")
    parser.add_argument("--quiet", action="store_true", help="suppress the metrics report")
    return parser

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        train(data_path=args.data, model_dir=args.model_dir, quiet=args.quiet,
              source=args.source)
    # RuntimeError covers the database side: a missing driver or a refused connection.
    except (FileNotFoundError, ValueError, RuntimeError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
