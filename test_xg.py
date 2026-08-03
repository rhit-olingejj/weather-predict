from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import xg
from conftest import CITIES, DAYS, TINY_GRID, make_raw

WARMUP = max(xg.ROLL_WINDOWS) - 1



@pytest.mark.parametrize("name, expected", [
    ("New York", "new_york"),
    ("São Paulo", "sao_paulo"),
    ("Ácme City", "acme_city"),
    ("TOKYO", "tokyo"),
    ("  spaced  ", "spaced"),
    ("Thunderstorm", "thunderstorm"),
])
def test_convert_names_folds_to_ascii_slug(name, expected):
    assert xg.convert_names(name) == expected


def test_convert_names_produces_xgboost_safe_identifiers():
    for city in ("St. John's [old]", "a<b", "Ünïcödé"):
        assert not set(xg.convert_names(city)) & set("[]<>")


@pytest.mark.parametrize("code, expected", [
    (0, "Clear"), (1, "Clear"),
    (3, "Cloudy"),
    (45, "Fog"),
    (53, "Drizzle"),
    (65, "Rain"), (82, "Rain"),
    (73, "Snow"),
    (95, "Thunderstorm"),
])
def test_categorize_wmo_known_codes(code, expected):
    assert xg.categorize_wmo(code) == expected

@pytest.mark.parametrize("code", [7, 99999, -1])
def test_categorize_wmo_unknown_code(code):
    assert xg.categorize_wmo(code) == xg.UNKNOWN_CATEGORY

def test_categorize_wmo_missing_value():
    assert xg.categorize_wmo(np.nan) == xg.UNKNOWN_CATEGORY
    assert xg.categorize_wmo(None) == xg.UNKNOWN_CATEGORY

def test_wmo_buckets_do_not_overlap():
    seen = [code for codes in xg.WMO_CATEGORIES.values() for code in codes]
    assert len(seen) == len(set(seen))


def test_load_raw_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_weather.py"):
        xg.load_raw(tmp_path / "nope.csv")


def test_load_raw_missing_columns(tmp_path):
    path = tmp_path / "partial.csv"
    pd.DataFrame({"city": ["A"], "date": ["2022-01-01"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing column"):
        xg.load_raw(path)


def test_load_raw_parses_sorts_and_dedupes(tmp_path):
    frame = make_raw(days=5)
    shuffled = pd.concat([frame.iloc[::-1], frame.head(3)])  # reversed + duplicates
    path = tmp_path / "messy.csv"
    shuffled.to_csv(path, index=False, encoding="utf-8")

    loaded = xg.load_raw(path)

    assert len(loaded) == len(frame)
    assert pd.api.types.is_datetime64_any_dtype(loaded["date"])
    assert loaded.equals(loaded.sort_values(["city", "date"]).reset_index(drop=True))

def test_labels_are_the_next_days_observations(raw, features):
    observed = raw.set_index(["city", "date"])
    labelled = features.dropna(subset=[xg.TARGET_TEMP])

    for _, row in labelled.sample(25, random_state=0).iterrows():
        tomorrow = observed.loc[(row["city"], row["target_date"])]
        assert row[xg.TARGET_TEMP] == pytest.approx(tomorrow["temp_mean_c"])
        assert row["y_precip_mm"] == pytest.approx(tomorrow["precip_mm"])
        assert row["y_condition"] == xg.categorize_wmo(tomorrow["wmo_code"])


def test_features_only_describe_today_or_earlier(raw, features):
    observed = raw.set_index(["city", "date"])
    for _, row in features.sample(25, random_state=1).iterrows():
        today = observed.loc[(row["city"], row["date"])]
        assert row["temp_mean_c"] == pytest.approx(today["temp_mean_c"])
        assert row["precip_mm"] == pytest.approx(today["precip_mm"])


def test_target_date_is_the_day_after(features):
    assert ((features["target_date"] - features["date"]) == pd.Timedelta(days=1)).all()


def test_last_day_of_each_city_has_no_labels(features):
    last = features.groupby("city")["date"].transform("max") == features["date"]
    assert features.loc[last, xg.TARGET_TEMP].isna().all()
    assert features.loc[last, xg.TARGET_PRECIP].isna().all()
    assert features.loc[last, "y_condition"].isna().all()


def test_gap_in_the_record_suppresses_labels(tmp_path):
    frame = make_raw(days=60, cities=CITIES[:1])
    gapped = frame.drop(frame.index[45]).reset_index(drop=True)
    path = tmp_path / "gapped.csv"
    gapped.to_csv(path, index=False, encoding="utf-8")

    built = xg.build_features(xg.load_raw(path))
    day_before_gap = built[built["date"] == pd.Timestamp(frame.loc[44, "date"])]

    assert len(day_before_gap) == 1
    assert pd.isna(day_before_gap.iloc[0][xg.TARGET_TEMP])
    assert pd.isna(day_before_gap.iloc[0][xg.TARGET_PRECIP])
    assert pd.isna(day_before_gap.iloc[0]["y_condition"])


def test_warmup_rows_are_dropped(raw, features):
    for city, group in features.groupby("city"):
        first_kept = group["date"].min()
        first_observed = raw.loc[raw["city"] == city, "date"].min()
        assert (first_kept - first_observed).days == WARMUP
        assert len(group) == DAYS - WARMUP


def test_no_nan_features_survive(features):
    assert not features[xg.feature_names(features)].isna().any().any()

def test_lag_and_rolling_values(raw, features):
    city = CITIES[0]["city"]
    history = raw[raw["city"] == city].reset_index(drop=True)
    row = features[features["city"] == city].iloc[10]
    position = history.index[history["date"] == row["date"]][0]

    assert row["temp_mean_c_lag1"] == pytest.approx(history.loc[position - 1, "temp_mean_c"])
    assert row["temp_mean_c_lag7"] == pytest.approx(history.loc[position - 7, "temp_mean_c"])

    window = history.loc[position - 6:position, "temp_mean_c"]
    assert row["temp_mean_c_roll7_mean"] == pytest.approx(window.mean())
    assert row["temp_delta_1d"] == pytest.approx(
        row["temp_mean_c"] - history.loc[position - 1, "temp_mean_c"])
    assert row["temp_anomaly_7d"] == pytest.approx(row["temp_mean_c"] - window.mean())


def test_wet_flag_uses_the_shared_threshold(features):
    expected = (features["precip_mm"] > xg.PRECIP_THRESHOLD_MM).astype(float)
    assert features["is_wet"].equals(expected)
    assert xg.PRECIP_THRESHOLD_MM == pytest.approx(0.1)


def test_precip_label_thresholds_next_day_rainfall(features):
    labelled = features.dropna(subset=["y_precip_mm"])
    expected = (labelled["y_precip_mm"] > xg.PRECIP_THRESHOLD_MM).astype(float)
    assert labelled[xg.TARGET_PRECIP].equals(expected)


def test_city_one_hots_are_exclusive(features):
    columns = [f"city_{xg.convert_names(c['city'])}" for c in CITIES]
    assert set(columns) <= set(features.columns)
    assert (features[columns].sum(axis=1) == 1).all()


def test_condition_one_hots_match_the_category(features):
    for category in xg.WMO_CATEGORIES:
        column = f"cond_today_{xg.convert_names(category)}"
        assert features[column].equals((features["condition"] == category).astype(float))


def test_hemisphere_flips_the_seasonal_wave(features):
    northern = features[features["city"] == "Testville"].set_index("date")
    southern = features[features["city"] == "Southtown"].set_index("date")
    shared = northern.index.intersection(southern.index)

    assert northern.loc[shared, "hemisphere_doy_sin"].to_numpy() == pytest.approx(
        -southern.loc[shared, "hemisphere_doy_sin"].to_numpy())


def test_seasonal_encoding_is_cyclical(features):
    unit = features["target_doy_sin"] ** 2 + features["target_doy_cos"] ** 2
    assert unit.to_numpy() == pytest.approx(1.0)

def test_feature_names_exclude_labels_and_bookkeeping(features):
    names = xg.feature_names(features)
    for column in (*xg.META_COLUMNS, *xg.TARGET_COLUMNS):
        assert column not in names


def test_feature_names_never_leak_tomorrow(features):
    names = xg.feature_names(features)
    assert not [name for name in names if name.startswith("y_")]
    assert "y_precip_mm" in features.columns  # present in the frame, just not a feature


def test_feature_names_are_all_numeric(features):
    frame = features[xg.feature_names(features)]
    assert all(pd.api.types.is_numeric_dtype(frame[c]) for c in frame.columns)


def test_todays_observations_are_kept_as_features(features):
    names = xg.feature_names(features)
    for column in ("temp_mean_c", "temp_max_c", "precip_mm", "wmo_code",
                   "latitude", "longitude"):
        assert column in names

def test_grid_entries_have_unique_names():
    names = [candidate["name"] for candidate in xg.PARAM_GRID]
    assert len(names) == len(set(names))


def test_grid_entries_share_one_shape():
    shapes = {frozenset(candidate) for candidate in xg.PARAM_GRID}
    assert len(shapes) == 1, "every candidate must set the same parameters"
    assert "n_estimators" in next(iter(shapes)), "tree budget is searched, not fixed"


def test_grid_and_fixed_params_do_not_collide():
    for candidate in xg.PARAM_GRID:
        assert not set(candidate) & set(xg.FIXED_PARAMS)


def test_selection_metrics_exist_in_the_scored_output(trained):
    for target, metric in xg.SELECTION_METRIC.items():
        assert metric in trained["bundle"]["metrics"][target]["validate"]


@pytest.fixture
def split(features):
    labelled = features.dropna(subset=[xg.TARGET_TEMP]).copy()
    names = xg.feature_names(labelled)
    cut = int(len(labelled) * 0.7)
    return (labelled[names].iloc[:cut], labelled[xg.TARGET_TEMP].iloc[:cut],
            labelled[names].iloc[cut:], labelled[xg.TARGET_TEMP].iloc[cut:])


def test_search_returns_trials_sorted_best_first(split):
    X_train, y_train, X_validate, y_validate = split
    trials = xg.search(xg.make_temperature_model,
                       lambda m, X, y: xg.evaluate_temperature(m, X, y, X["temp_mean_c"]),
                       "mae_c", X_train, y_train, X_validate, y_validate,
                       grid=TINY_GRID, say=lambda *_: None)

    assert len(trials) == len(TINY_GRID)
    assert [t["score"] for t in trials] == sorted(t["score"] for t in trials)
    assert {t["name"] for t in trials} == {c["name"] for c in TINY_GRID}


def test_search_flags_a_budget_bound_candidate(split):
    X_train, y_train, X_validate, y_validate = split
    starved = ({**TINY_GRID[0], "name": "starved", "n_estimators": 3},)

    trial = xg.search(xg.make_temperature_model,
                      lambda m, X, y: xg.evaluate_temperature(m, X, y, X["temp_mean_c"]),
                      "mae_c", X_train, y_train, X_validate, y_validate,
                      grid=starved, say=lambda *_: None)[0]

    assert trial["capped"] is True
    assert trial["best_iteration"] + 1 <= 3


def test_search_reports_progress(split):
    X_train, y_train, X_validate, y_validate = split
    lines = []
    xg.search(xg.make_temperature_model,
              lambda m, X, y: xg.evaluate_temperature(m, X, y, X["temp_mean_c"]),
              "mae_c", X_train, y_train, X_validate, y_validate,
              grid=TINY_GRID, say=lines.append)

    assert len(lines) == len(TINY_GRID)
    assert all("mae_c" in line for line in lines)

def test_learning_curve_is_ordered_and_ends_at_the_stopping_round(split):
    X_train, y_train, X_validate, y_validate = split
    model = xg.make_temperature_model(
        {k: v for k, v in TINY_GRID[1].items() if k != "name"})
    model.fit(X_train, y_train, eval_set=[(X_validate, y_validate)], verbose=False)

    curve = xg.learning_curve(model)

    assert curve
    rounds = [r for r, _ in curve]
    assert rounds == sorted(rounds)
    assert len(rounds) == len(set(rounds))
    assert model.best_iteration + 1 in rounds

def test_train_bundles_a_model_per_target(trained):
    bundle = trained["bundle"]
    for key in ("temp_model", "precip_model", "condition_model", "feature_names",
                "condition_classes", "cities", "metrics", "trained_at", "split"):
        assert key in bundle


def test_train_writes_the_bundle_to_disk(trained):
    assert (trained["model_dir"] / xg.MODEL_FILENAME).exists()


def test_train_records_every_trial(trained):
    for target, entry in trained["bundle"]["metrics"].items():
        assert len(entry["trials"]) == len(TINY_GRID)
        assert entry["config"] in {c["name"] for c in TINY_GRID}
        assert "model" not in entry["trials"][0], "models must not bloat the bundle"


def test_train_keeps_the_best_scoring_candidate(trained):
    for entry in trained["bundle"]["metrics"].values():
        best = min(trial["score"] for trial in entry["trials"])
        assert entry["trials"][0]["score"] == pytest.approx(best)
        assert entry["config"] == entry["trials"][0]["name"]


def test_train_scores_both_validate_and_test(trained):
    for entry in trained["bundle"]["metrics"].values():
        assert entry["validate"] and entry["test"]
        assert set(entry["validate"]) == set(entry["test"])


def test_condition_classes_match_the_synthetic_data(trained):
    assert set(trained["bundle"]["condition_classes"]) == {"Clear", "Cloudy",
                                                           "Drizzle", "Rain"}


def test_bundle_feature_names_match_the_fitted_models(trained):
    bundle = trained["bundle"]
    for key in ("temp_model", "precip_model", "condition_model"):
        assert list(bundle[key].feature_names_in_) == bundle["feature_names"]
