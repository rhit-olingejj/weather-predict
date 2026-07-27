from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import predict_weather as pw
import xg

KNOWN = ["London", "New York", "Sydney", "São Paulo", "Tokyo"]

@pytest.mark.parametrize("query, expected", [
    ("New York", "New York"),
    ("new york", "New York"),
    ("NEW YORK", "New York"),
    ("new_york", "New York"),
    ("São Paulo", "São Paulo"),
    ("sao paulo", "São Paulo"),
    ("tok", "Tokyo"),
    ("lond", "London"),
])
def test_match_city_resolves_to_canonical(query, expected):
    assert pw.match_city(query, KNOWN) == expected


def test_match_city_rejects_unknown():
    with pytest.raises(ValueError, match="unknown city"):
        pw.match_city("Atlantis", KNOWN)


def test_match_city_error_lists_the_options():
    with pytest.raises(ValueError) as err:
        pw.match_city("Atlantis", KNOWN)
    for city in KNOWN:
        assert city in str(err.value)


def test_match_city_does_not_collapse_repeated_separators():
    """Documents a known limit: "Sao  Paulo" folds to sao__paulo and misses."""
    with pytest.raises(ValueError):
        pw.match_city("Sao  Paulo", KNOWN)


def test_match_city_rejects_ambiguous_prefix():
    with pytest.raises(ValueError):
        pw.match_city("s", ["Sydney", "São Paulo"])


def test_match_city_prefers_exact_over_prefix():
    assert pw.match_city("San", ["San", "San Diego"]) == "San"

@pytest.mark.parametrize("value, kind", [
    (np.int64(3), int),
    (np.float32(1.5), float),
    (np.bool_(True), bool),
])
def test_to_builtin_unwraps_numpy(value, kind):
    result = pw._to_builtin(value)
    assert type(result) is kind
    json.dumps(result)  # must survive serialization


def test_to_builtin_passes_through_python_values():
    for value in ("text", 5, 2.5, True, None):
        assert pw._to_builtin(value) is value

def test_load_missing_bundle(tmp_path):
    with pytest.raises(FileNotFoundError, match="xg.py"):
        pw.WeatherPredictor.load(tmp_path)


def test_load_round_trips_a_trained_bundle(trained):
    predictor = pw.WeatherPredictor.load(trained["model_dir"])
    assert predictor.feature_names == trained["bundle"]["feature_names"]
    assert predictor.condition_classes == trained["bundle"]["condition_classes"]


def test_bundle_keys_match_the_constructor(trained):
    pw.WeatherPredictor(**trained["bundle"])

@pytest.fixture
def predictor(trained):
    loaded = pw.WeatherPredictor.load(trained["model_dir"])
    loaded.data_path = str(trained["data_path"])
    return loaded


@pytest.fixture
def forecast(predictor):
    return predictor.predict("Testville", "2022-05-01")


def test_predict_targets_the_following_day(forecast):
    assert forecast["as_of_date"] == "2022-05-01"
    assert forecast["target_date"] == "2022-05-02"


def test_predict_returns_the_documented_shape(forecast):
    assert set(forecast) == {
        "city", "as_of_date", "target_date", "location", "temperature",
        "precipitation", "condition", "observed_on_as_of_date", "model",
    }
    assert set(forecast["temperature"]) == {"mean_c", "mean_f", "expected_error_c"}
    assert set(forecast["precipitation"]) == {"probability", "will_precipitate",
                                              "threshold_mm"}
    assert set(forecast["condition"]) == {"category", "confidence", "probabilities"}


def test_predict_output_is_json_serializable(forecast):
    assert json.loads(json.dumps(forecast)) == forecast


def test_temperature_units_agree(forecast):
    # Both are converted from the full-precision prediction and rounded
    # separately, so mean_c's own rounding can drift them by up to 0.05 * 9/5.
    celsius = forecast["temperature"]["mean_c"]
    assert forecast["temperature"]["mean_f"] == pytest.approx(celsius * 9 / 5 + 32, abs=0.1)


def test_precipitation_probability_is_a_probability(forecast):
    probability = forecast["precipitation"]["probability"]
    assert 0.0 <= probability <= 1.0
    assert forecast["precipitation"]["will_precipitate"] is (probability >= 0.5)
    assert forecast["precipitation"]["threshold_mm"] == xg.PRECIP_THRESHOLD_MM


def test_condition_distribution_is_normalised(forecast, predictor):
    probabilities = forecast["condition"]["probabilities"]
    assert set(probabilities) == set(predictor.condition_classes)
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-3)


def test_condition_reports_the_most_likely_class(forecast):
    probabilities = forecast["condition"]["probabilities"]
    winner = max(probabilities, key=probabilities.get)
    assert forecast["condition"]["category"] == winner
    assert forecast["condition"]["confidence"] == pytest.approx(probabilities[winner])


def test_condition_probabilities_are_sorted_descending(forecast):
    values = list(forecast["condition"]["probabilities"].values())
    assert values == sorted(values, reverse=True)


def test_observed_block_reflects_the_as_of_day(forecast, trained):
    raw = xg.load_raw(trained["data_path"])
    actual = raw[(raw["city"] == "Testville")
                 & (raw["date"] == pd.Timestamp("2022-05-01"))].iloc[0]
    observed = forecast["observed_on_as_of_date"]
    assert observed["temp_mean_c"] == pytest.approx(actual["temp_mean_c"])
    assert observed["precip_mm"] == pytest.approx(actual["precip_mm"])
    assert observed["condition"] == xg.categorize_wmo(actual["wmo_code"])


def test_predict_accepts_accented_and_lowercase_names(predictor):
    for query in ("Ácme City", "acme city", "ACME CITY"):
        assert predictor.predict(query, "2022-05-01")["city"] == "Ácme City"


def test_predict_accepts_timestamps_and_dates(predictor):
    from datetime import date

    expected = predictor.predict("Testville", "2022-05-01")["target_date"]
    for value in (pd.Timestamp("2022-05-01"), date(2022, 5, 1),
                  pd.Timestamp("2022-05-01 18:30")):
        assert predictor.predict("Testville", value)["target_date"] == expected


def test_predict_is_deterministic(predictor):
    first = predictor.predict("Testville", "2022-05-01")
    second = predictor.predict("Testville", "2022-05-01")
    assert first == second


def test_predict_rejects_unknown_city(predictor):
    with pytest.raises(ValueError, match="unknown city"):
        predictor.predict("Atlantis", "2022-05-01")


def test_predict_rejects_a_date_outside_the_record(predictor):
    with pytest.raises(ValueError, match="no usable history"):
        predictor.predict("Testville", "1999-01-01")


def test_predict_rejects_warmup_dates(predictor):
    """The first 29 days per city have no full 30-day window."""
    with pytest.raises(ValueError, match="no usable history"):
        predictor.predict("Testville", "2022-01-05")


def test_predict_error_names_the_usable_range(predictor):
    with pytest.raises(ValueError) as err:
        predictor.predict("Testville", "1999-01-01")
    assert "2022-01-30" in str(err.value)


def test_predict_serves_the_final_observed_day(predictor, trained):
    """The last day has no label but is still forecastable."""
    last = xg.load_raw(trained["data_path"])["date"].max()
    forecast = predictor.predict("Testville", last)
    assert forecast["target_date"] == (last + pd.Timedelta(days=1)).date().isoformat()


def test_predict_uses_only_that_citys_row(predictor):
    a = predictor.predict("Testville", "2022-05-01")
    b = predictor.predict("Southtown", "2022-05-01")
    assert a["city"] != b["city"]
    assert a["observed_on_as_of_date"] != b["observed_on_as_of_date"]


def test_predict_batch_matches_individual_calls(predictor):
    requests = [("Testville", "2022-05-01"), ("Southtown", "2022-05-15")]
    assert predictor.predict_batch(requests) == [
        predictor.predict(city, date) for city, date in requests]


def test_known_cities_are_sorted(predictor):
    assert predictor.known_cities == sorted(predictor.known_cities)


def test_expected_error_comes_from_the_test_split(predictor):
    reported = predictor.predict("Testville", "2022-05-01")["temperature"]["expected_error_c"]
    assert reported == pytest.approx(
        predictor.metrics["temperature"]["test"]["mae_c"], abs=0.005)


def test_format_prediction_mentions_the_key_numbers(forecast):
    rendered = pw.format_prediction(forecast)
    assert forecast["city"] in rendered
    assert forecast["target_date"] in rendered
    assert forecast["condition"]["category"] in rendered
    assert f"{forecast['temperature']['mean_c']:.1f} C" in rendered


def test_format_prediction_handles_a_single_condition_class(forecast):
    """No runner-up line when there is nothing to run up against."""
    only_one = {**forecast, "condition": {**forecast["condition"],
                                          "probabilities": {"Rain": 1.0}}}
    assert "also:" not in pw.format_prediction(only_one)


def test_cli_prints_a_forecast(trained, capsys):
    code = pw.main(["--city", "Testville", "--date", "2022-05-01",
                    "--model-dir", str(trained["model_dir"]),
                    "--data", str(trained["data_path"])])
    assert code == 0
    assert "Forecast for Testville on 2022-05-02" in capsys.readouterr().out


def test_cli_emits_json(trained, capsys):
    code = pw.main(["--city", "Testville", "--date", "2022-05-01", "--json",
                    "--model-dir", str(trained["model_dir"]),
                    "--data", str(trained["data_path"])])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["city"] == "Testville"


def test_cli_reports_errors_without_a_traceback(trained, capsys):
    code = pw.main(["--city", "Atlantis", "--date", "2022-05-01",
                    "--model-dir", str(trained["model_dir"]),
                    "--data", str(trained["data_path"])])
    assert code == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("error:")
    assert not captured.out
