#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import (
    DEFAULT_DATA,
    DEFAULT_MODEL_DIR,
    MODEL_FILENAME,
    PRECIP_THRESHOLD_MM,
)
from xg import build_features, convert_names, load_raw

API_VERSION = "1.0"


def match_city(query: str, known: list[str]) -> str:
    wanted = convert_names(query)
    for name in known:
        if convert_names(name) == wanted:
            return name
    partial = [name for name in known if convert_names(name).startswith(wanted)]
    if len(partial) == 1:
        return partial[0]
    raise ValueError(
        f"unknown city {query!r}; expected one of: {', '.join(sorted(known))}"
    )


def _to_builtin(value):
    """Make numpy scalars JSON-serializable."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


class WeatherPredictor:
    def __init__(self, temp_model, precip_model, condition_model,
                 feature_names: list[str], condition_classes: list[str],
                 cities: dict, data_path: str, split: str, metrics: dict,
                 trained_at: str, bundle_version: str = "unknown"):
        self.temp_model = temp_model
        self.precip_model = precip_model
        self.condition_model = condition_model
        self.feature_names = feature_names
        self.condition_classes = condition_classes
        self.cities = cities
        self.data_path = data_path
        self.split = split
        self.metrics = metrics
        self.trained_at = trained_at
        self.bundle_version = bundle_version
        self._features: pd.DataFrame | None = None

    @classmethod
    def load(cls, model_dir: str | Path = DEFAULT_MODEL_DIR) -> "WeatherPredictor":
        path = Path(model_dir) / MODEL_FILENAME
        if not path.exists():
            raise FileNotFoundError(f"{path} not found -- run `python xg.py` first")
        return cls(**joblib.load(path))

    @property
    def known_cities(self) -> list[str]:
        return sorted(self.cities)

    def features(self, data_path: str | Path | None = None) -> pd.DataFrame:
        if self._features is None or data_path is not None:
            self._features = build_features(load_raw(data_path or self.data_path))
        return self._features

    def predict(self, city: str, date, data_path: str | Path | None = None) -> dict:
        features = self.features(data_path)
        canonical = match_city(city, sorted(features["city"].unique()))
        as_of = pd.Timestamp(date).normalize()

        rows = features[(features["city"] == canonical) & (features["date"] == as_of)]
        if rows.empty:
            available = features.loc[features["city"] == canonical, "date"]
            raise ValueError(
                f"no usable history for {canonical} on {as_of.date()}; "
                f"available dates run {available.min().date()} to {available.max().date()} "
                f"(the first 29 days of each city are consumed as warm-up)"
            )
        row = rows.iloc[[-1]]
        X = row[self.feature_names]

        temperature = float(self.temp_model.predict(X)[0])
        precip_probability = float(self.precip_model.predict_proba(X)[0, 1])
        condition_probabilities = self.condition_model.predict_proba(X)[0]
        best = int(condition_probabilities.argmax())

        observed = row.iloc[0]
        return {
            "city": canonical,
            "as_of_date": as_of.date().isoformat(),
            "target_date": (as_of + pd.Timedelta(days=1)).date().isoformat(),
            "location": self.cities.get(canonical, {
                "latitude": _to_builtin(observed.get("latitude")),
                "longitude": _to_builtin(observed.get("longitude")),
            }),
            "temperature": {
                "mean_c": round(temperature, 1),
                "mean_f": round(temperature * 9 / 5 + 32, 1),
                "expected_error_c": round(self._test_metric("temperature", "mae_c"), 2),
            },
            "precipitation": {
                "probability": round(precip_probability, 4),
                "will_precipitate": bool(precip_probability >= 0.5),
                "threshold_mm": PRECIP_THRESHOLD_MM,
            },
            "condition": {
                "category": self.condition_classes[best],
                "confidence": round(float(condition_probabilities[best]), 4),
                "probabilities": {
                    name: round(float(probability), 4)
                    for name, probability in sorted(
                        zip(self.condition_classes, condition_probabilities),
                        key=lambda pair: pair[1],
                        reverse=True,
                    )
                },
            },
            "observed_on_as_of_date": {
                "temp_mean_c": _to_builtin(observed["temp_mean_c"]),
                "temp_max_c": _to_builtin(observed["temp_max_c"]),
                "temp_min_c": _to_builtin(observed["temp_min_c"]),
                "precip_mm": _to_builtin(observed["precip_mm"]),
                "condition": observed["condition"],
            },
            "model": {
                "api_version": API_VERSION,
                "bundle_version": self.bundle_version,
                "trained_at": self.trained_at,
                "split": self.split,
            },
        }

    def predict_batch(self, requests: list[tuple[str, str]]) -> list[dict]:
        return [self.predict(city, date) for city, date in requests]

    def _test_metric(self, target: str, name: str) -> float:
        return self.metrics.get(target, {}).get("test", {}).get(name, float("nan"))


def format_prediction(prediction: dict) -> str:
    temperature = prediction["temperature"]
    precipitation = prediction["precipitation"]
    condition = prediction["condition"]
    observed = prediction["observed_on_as_of_date"]

    lines = [
        "",
        f"Forecast for {prediction['city']} on {prediction['target_date']} "
        f"(issued from {prediction['as_of_date']})",
        "-" * 68,
        f"  Temperature   {temperature['mean_c']:.1f} C / {temperature['mean_f']:.1f} F"
        f"  (+/- {temperature['expected_error_c']:.2f} C typical error)",
        f"  Precipitation {precipitation['probability']:.0%} chance of "
        f"> {precipitation['threshold_mm']} mm"
        f"  -> {'yes' if precipitation['will_precipitate'] else 'no'}",
        f"  Condition     {condition['category']} "
        f"({condition['confidence']:.0%} confident)",
    ]
    runners_up = list(condition["probabilities"].items())[1:4]
    if runners_up:
        lines.append("                also: "
                     + ", ".join(f"{name} {p:.0%}" for name, p in runners_up))
    lines.append(f"  Conditioned on {prediction['as_of_date']}: "
                 f"{observed['temp_mean_c']} C, {observed['precip_mm']} mm, "
                 f"{observed['condition']}")
    return "\n".join(lines)


#  CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--city", required=True, help='e.g. "New York", tokyo, "sao paulo"')
    parser.add_argument("--date", required=True,
                        help="last observed day, YYYY-MM-DD; forecast is for the next day")
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help=f"observation CSV (default: {DEFAULT_DATA})")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                        help=f"where the model bundle lives (default: {DEFAULT_MODEL_DIR})")
    parser.add_argument("--json", action="store_true",
                        help="emit the structured result as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        predictor = WeatherPredictor.load(args.model_dir)
        prediction = predictor.predict(args.city, args.date, data_path=args.data)
    except (FileNotFoundError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    print(json.dumps(prediction, indent=2) if args.json else format_prediction(prediction))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
