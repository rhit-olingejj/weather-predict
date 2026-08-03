from __future__ import annotations

import sys
from unittest import mock

import pandas as pd
import pytest

import db
import xg
from conftest import make_raw

# The columns db.OBSERVATION_QUERY selects, in order -- a fake cursor reports these as its description so the loader is exercised the way psycopg2 would drive it.
QUERY_COLUMNS = ("city", "latitude", "longitude", "date", "temp_mean_c", "temp_max_c",
                 "temp_min_c", "precip_mm", "wmo_code")

ENV = {
    "DB_HOST": "db.internal",
    "DB_PORT": "6543",
    "DB_NAME": "weather_predict",
    "DB_USER": "app",
    "DB_PASSWORD": "s3cret",
    "DB_SCHEMA": "weather_predict_db",
}


class FakeCursor:
    def __init__(self, rows, columns=QUERY_COLUMNS):
        self._rows = rows
        self.description = [(name,) for name in columns]
        self.executed: list[tuple[str, dict]] = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, query, params=None):
        self.executed.append((query, params or {}))

    def fetchall(self):
        return self._rows


class FakeConnection:
    def __init__(self, rows, columns=QUERY_COLUMNS):
        self.cursors = [FakeCursor(rows, columns)]
        self.closed = False

    def cursor(self):
        return self.cursors[-1]

    def close(self):
        self.closed = True


def rows_from(frame: pd.DataFrame) -> list[tuple]:
    return list(frame[list(QUERY_COLUMNS)].itertuples(index=False, name=None))


def test_config_reads_every_setting_from_the_environment():
    config = db.DbConfig.from_env(ENV)

    assert (config.host, config.port, config.name) == ("db.internal", 6543, "weather_predict")
    assert (config.user, config.password) == ("app", "s3cret")
    assert config.schema == "weather_predict_db"


def test_config_falls_back_to_defaults():
    config = db.DbConfig.from_env({"DB_USER": "app"})

    assert config.host == db.DEFAULT_HOST
    assert config.port == db.DEFAULT_PORT
    assert config.schema == db.DEFAULT_SCHEMA
    assert config.timezone == db.DEFAULT_TIMEZONE
    assert config.min_obs_per_day == db.DEFAULT_MIN_OBS_PER_DAY


def test_config_requires_a_user_when_no_url_is_given():
    with pytest.raises(ValueError, match="DB_USER"):
        db.DbConfig.from_env({"DB_HOST": "localhost", "DB_NAME": "weather_predict"})


def test_config_accepts_a_url_instead_of_the_pieces():
    config = db.DbConfig.from_env(
        {"DATABASE_URL": "postgresql://app:s3cret@db.internal:6543/weather_predict"})
    assert config.url.endswith("/weather_predict")


def test_config_rejects_a_non_integer_port():
    with pytest.raises(ValueError, match="DB_PORT"):
        db.DbConfig.from_env({**ENV, "DB_PORT": "not-a-port"})


def test_config_rejects_a_schema_that_is_not_an_identifier():
    with pytest.raises(ValueError, match="DB_SCHEMA"):
        db.DbConfig.from_env({**ENV, "DB_SCHEMA": 'public"; DROP TABLE cities --'})


@pytest.mark.parametrize("env", [ENV, {"DATABASE_URL": "postgresql://app:s3cret@h/db"}])
def test_describe_never_leaks_the_password(env):
    described = db.DbConfig.from_env(env).describe()
    assert "s3cret" not in described


def test_is_configured_only_when_credentials_are_present():
    assert db.is_configured(ENV)
    assert db.is_configured({"DATABASE_URL": "postgresql://app@h/db"})
    assert not db.is_configured({"DB_HOST": "localhost"})


def test_query_targets_the_configured_schema():
    config = db.DbConfig.from_env({**ENV, "DB_SCHEMA": "wx_staging"})
    connection = FakeConnection(rows_from(make_raw(days=3)))

    db.fetch_observations(config, connection=connection)
    query, params = connection.cursors[-1].executed[0]

    assert "wx_staging.wx_data" in query and "wx_staging.cities" in query
    assert params == {"timezone": config.timezone, "min_obs": config.min_obs_per_day}


def test_fetch_returns_the_columns_training_needs():
    frame = db.fetch_observations(db.DbConfig.from_env(ENV),
                                  connection=FakeConnection(rows_from(make_raw(days=4))))

    assert set(xg.REQUIRED_COLUMNS) <= set(frame.columns)
    assert len(frame) == 4 * 3  # days x cities in the fixture


def test_fetch_rejects_an_empty_result():
    with pytest.raises(ValueError, match="wx_data"):
        db.fetch_observations(db.DbConfig.from_env(ENV), connection=FakeConnection([]))


def test_fetch_closes_a_connection_it_opened():
    connection = FakeConnection(rows_from(make_raw(days=3)))
    with mock.patch.object(db, "connect", return_value=connection):
        db.fetch_observations(db.DbConfig.from_env(ENV))
    assert connection.closed


def test_missing_driver_is_reported_as_a_runtime_error():
    # A None entry in sys.modules makes `import psycopg2` raise ImportError whether or not the driver happens to be installed here.
    with mock.patch.dict(sys.modules, {"psycopg2": None}):
        with pytest.raises(RuntimeError, match="psycopg2"):
            db.connect(db.DbConfig.from_env(ENV))


def test_database_rows_normalize_into_the_same_frame_as_the_csv(tmp_path):
    frame = make_raw(days=40)
    path = tmp_path / "weather_daily.csv"
    frame.to_csv(path, index=False, encoding="utf-8")

    from_csv = xg.load_raw(path)
    from_db = xg.normalize_raw(
        db.fetch_observations(db.DbConfig.from_env(ENV),
                              connection=FakeConnection(rows_from(frame))),
        "test database")

    pd.testing.assert_frame_equal(from_db[list(from_csv.columns)], from_csv,
                                  check_dtype=False)


def test_database_rows_build_the_same_features_as_the_csv(tmp_path):
    frame = make_raw(days=40)
    path = tmp_path / "weather_daily.csv"
    frame.to_csv(path, index=False, encoding="utf-8")

    from_db = xg.normalize_raw(
        db.fetch_observations(db.DbConfig.from_env(ENV),
                              connection=FakeConnection(rows_from(frame))),
        "test database")

    csv_features = xg.build_features(xg.load_raw(path))
    db_features = xg.build_features(from_db)

    assert list(db_features.columns) == list(csv_features.columns)
    pd.testing.assert_frame_equal(db_features[xg.feature_names(db_features)],
                                  csv_features[xg.feature_names(csv_features)],
                                  check_dtype=False)


def test_two_cities_sharing_a_name_are_refused_not_deduplicated():
    frame = make_raw(days=5)
    twin = frame[frame["city"] == "Testville"].copy()
    twin["latitude"] = -20.0  # a different city that happens to share the name
    doubled = db.fetch_observations(
        db.DbConfig.from_env(ENV),
        connection=FakeConnection(rows_from(pd.concat([frame, twin]))))

    with mock.patch.object(db, "fetch_observations", return_value=doubled):
        with mock.patch.dict("os.environ", ENV, clear=True):
            with pytest.raises(ValueError, match="Testville"):
                xg.load_db()


def test_load_observations_dispatches_to_the_database():
    with mock.patch.object(xg, "load_db", return_value=make_raw(days=3)) as loader:
        xg.load_observations("db", "ignored.csv")
    loader.assert_called_once_with()


def test_load_observations_rejects_an_unknown_source():
    with pytest.raises(ValueError, match="unknown data source"):
        xg.load_observations("mysql", "weather_daily.csv")


def test_normalize_raw_reports_missing_columns():
    with pytest.raises(ValueError, match="missing column"):
        xg.normalize_raw(pd.DataFrame({"city": ["A"], "date": ["2022-01-01"]}), "db")
