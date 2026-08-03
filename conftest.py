from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest

import xg

# Three cities: one northern, one southern (to exercise the hemisphere-signed seasonal feature), and one accented (to exercise name folding).
CITIES = (
    {"city": "Testville", "latitude": 40.0, "longitude": -74.0},
    {"city": "Southtown", "latitude": -33.0, "longitude": 151.0},
    {"city": "Ácme City", "latitude": 10.0, "longitude": 20.0},
)
DAYS = 150
START = "2022-01-01"

# Cycled so both wet and dry days occur and at least three condition categories appear: 0 -> Clear, 3 -> Cloudy, 51 -> Drizzle, 61 -> Rain.
WMO_CYCLE = (0, 3, 51, 61, 3, 0, 61)

# A grid small enough to train in seconds, still exercising both the budget-bound and early-stopped paths.
TINY_GRID = (
    {"name": "tiny-a", "n_estimators": 8, "max_depth": 2, "learning_rate": 0.3,
     "min_child_weight": 1, "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 1.0},
    {"name": "tiny-b", "n_estimators": 12, "max_depth": 3, "learning_rate": 0.2,
     "min_child_weight": 2, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 2.0},
)


def make_raw(days: int = DAYS, cities=CITIES) -> pd.DataFrame:
    dates = pd.date_range(START, periods=days, freq="D")
    rows = []
    for offset, city in enumerate(cities):
        for i, day in enumerate(dates):
            season = 10.0 * np.sin(2 * np.pi * i / 365.25)
            mean = 15.0 + offset * 3 + season + (i % 5) * 0.4
            rows.append({
                **city,
                "date": day.strftime("%Y-%m-%d"),
                "temp_mean_c": round(mean, 1),
                "temp_max_c": round(mean + 4 + (i % 3), 1),
                "temp_min_c": round(mean - 4 - (i % 2), 1),
                "precip_mm": round((i % 4) * 1.5, 1),
                "wmo_code": WMO_CYCLE[(i + offset) % len(WMO_CYCLE)],
            })
    return pd.DataFrame(rows)


@pytest.fixture
def raw_csv(tmp_path):
    path = tmp_path / "weather_daily.csv"
    make_raw().to_csv(path, index=False, encoding="utf-8")
    return path


@pytest.fixture
def raw(raw_csv):
    return xg.load_raw(raw_csv)


@pytest.fixture
def features(raw):
    return xg.build_features(raw)


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    directory = tmp_path_factory.mktemp("trained")
    csv_path = directory / "weather_daily.csv"
    make_raw().to_csv(csv_path, index=False, encoding="utf-8")

    # source is pinned rather than left to DATA_SOURCE: a developer with a filled-in .env must not have the suite reach for their database.
    with mock.patch.object(xg, "PARAM_GRID", TINY_GRID):
        bundle = xg.train(data_path=str(csv_path), model_dir=str(directory), quiet=True,
                          source="csv")
    return {"bundle": bundle, "model_dir": directory, "data_path": csv_path}
