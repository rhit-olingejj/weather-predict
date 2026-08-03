#!/usr/bin/env python3
"""Read daily observations out of the weather_predict SQL database (PostgreSQL).

Connection details come from environment variables -- normally a local `.env`
(copy `.env.example`), loaded by `config.py` so every entry point sees them:

    DATABASE_URL          full libpq URL; wins over the pieces below if set
    DB_HOST DB_PORT DB_NAME DB_USER DB_PASSWORD
    DB_SCHEMA             default weather_predict_db
    DB_SSLMODE            libpq sslmode, default prefer
    DB_TIMEZONE           timezone the day boundary is drawn in, default UTC
    DB_MIN_OBS_PER_DAY    drop city-days with fewer temperature readings

`wx_data` holds one row per observation timestamp (`dtg`) while training wants
one row per city-day, so OBSERVATION_QUERY does the rollup: mean/max/min of
`temp_C`, the day's summed `precip_mm`, and its highest WMO code. Given hourly
rows that recovers a true daily range; if the database only holds one row per
day then max == min == mean and `temp_range_c` collapses to zero -- the models
still train, they just lose that feature.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Mapping

import pandas as pd

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 5432
DEFAULT_NAME = "weather_predict"
DEFAULT_SCHEMA = "weather_predict_db"
DEFAULT_SSLMODE = "prefer"
DEFAULT_TIMEZONE = "UTC"
DEFAULT_MIN_OBS_PER_DAY = 1

ENV_EXAMPLE = ".env.example"

# Schema names arrive from the environment and cannot travel as bind parameters, so they are checked against a plain identifier before being interpolated into SQL.
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# The day's WMO code is taken as MAX rather than the modal value: codes in WMO 4677 climb roughly with severity, so the maximum is the day's most significant weather -- the same convention Open-Meteo's daily weather_code follows.
OBSERVATION_QUERY = """
SELECT c.city_name                              AS city,
       c.latitude::float8                       AS latitude,
       c.longitude::float8                      AS longitude,
       (w.dtg AT TIME ZONE %(timezone)s)::date  AS "date",
       AVG(w.temp_C)::float8                    AS temp_mean_c,
       MAX(w.temp_C)::float8                    AS temp_max_c,
       MIN(w.temp_C)::float8                    AS temp_min_c,
       SUM(w.precip_mm)::float8                 AS precip_mm,
       MAX(NULLIF(TRIM(w.wmo_code), '')::int)   AS wmo_code
  FROM {schema}.wx_data w
  -- Grouped by city_id as well as name: city names are only unique within an admin division, and two same-named cities must not average together.
  JOIN {schema}.cities c ON c.city_id = w.city_id
 GROUP BY c.city_id, c.city_name, c.latitude, c.longitude,
          (w.dtg AT TIME ZONE %(timezone)s)::date
HAVING COUNT(w.temp_C) >= %(min_obs)s
 ORDER BY c.city_name, "date"
"""


def _text(env: Mapping[str, str], key: str, default: str = "") -> str:
    return str(env.get(key) or default).strip()


def _int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = _text(env, key)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as err:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from err


def _redact(url: str) -> str:
    return re.sub(r"://([^:/@]+):[^@]*@", r"://\1:***@", url)


@dataclass(frozen=True)
class DbConfig:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    name: str = DEFAULT_NAME
    user: str = ""
    password: str = ""
    schema: str = DEFAULT_SCHEMA
    sslmode: str = DEFAULT_SSLMODE
    timezone: str = DEFAULT_TIMEZONE
    min_obs_per_day: int = DEFAULT_MIN_OBS_PER_DAY
    url: str = ""

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "DbConfig":
        env = os.environ if env is None else env

        url = _text(env, "DATABASE_URL")
        host = _text(env, "DB_HOST", DEFAULT_HOST)
        name = _text(env, "DB_NAME", DEFAULT_NAME)
        user = _text(env, "DB_USER")

        # DB_PASSWORD may legitimately be blank (peer auth, ~/.pgpass), so it is not required here.
        if not url:
            missing = [key for key, value in (("DB_HOST", host), ("DB_NAME", name),
                                              ("DB_USER", user)) if not value]
            if missing:
                raise ValueError(
                    f"database source requested but {', '.join(missing)} unset -- "
                    f"copy {ENV_EXAMPLE} to .env and fill it in"
                )

        schema = _text(env, "DB_SCHEMA", DEFAULT_SCHEMA)
        if not IDENTIFIER.match(schema):
            raise ValueError(f"DB_SCHEMA {schema!r} is not a plain SQL identifier")

        return cls(
            host=host,
            port=_int(env, "DB_PORT", DEFAULT_PORT),
            name=name,
            user=user,
            password=str(env.get("DB_PASSWORD") or ""),
            schema=schema,
            sslmode=_text(env, "DB_SSLMODE", DEFAULT_SSLMODE),
            timezone=_text(env, "DB_TIMEZONE", DEFAULT_TIMEZONE),
            min_obs_per_day=_int(env, "DB_MIN_OBS_PER_DAY", DEFAULT_MIN_OBS_PER_DAY),
            url=url,
        )

    def describe(self) -> str:
        """Where the rows came from, safe to print -- never includes the password."""
        if self.url:
            return f"{_redact(self.url)} ({self.schema})"
        return f"{self.user}@{self.host}:{self.port}/{self.name} ({self.schema})"


def is_configured(env: Mapping[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    return bool(_text(env, "DATABASE_URL") or _text(env, "DB_USER"))


def connect(config: DbConfig | None = None):
    """Open a psycopg2 connection. The caller closes it."""
    config = DbConfig.from_env() if config is None else config
    try:
        import psycopg2
    except ImportError as err:  # only the database source needs the driver
        raise RuntimeError(
            "psycopg2 is required to read from the database -- "
            "pip install -r requirements.txt"
        ) from err

    try:
        if config.url:
            return psycopg2.connect(config.url)
        return psycopg2.connect(host=config.host, port=config.port, dbname=config.name,
                                user=config.user, password=config.password,
                                sslmode=config.sslmode)
    except psycopg2.Error as err:
        raise RuntimeError(f"could not connect to {config.describe()}: {err}") from err


def _run(connection, query: str, params: dict) -> pd.DataFrame:
    with connection.cursor() as cursor:
        cursor.execute(query, params)
        columns = [column[0] for column in cursor.description]
        rows = cursor.fetchall()
    # Built by hand rather than with read_sql: pandas wants a SQLAlchemy connectable and warns on a raw DB-API connection.
    return pd.DataFrame(rows, columns=columns)


def fetch_observations(config: DbConfig | None = None, connection=None) -> pd.DataFrame:
    """One row per city-day, in the same shape fetch_weather.py writes to CSV."""
    config = DbConfig.from_env() if config is None else config
    query = OBSERVATION_QUERY.format(schema=config.schema)
    params = {"timezone": config.timezone, "min_obs": config.min_obs_per_day}

    if connection is not None:
        frame = _run(connection, query, params)
    else:
        open_connection = connect(config)
        try:
            frame = _run(open_connection, query, params)
        finally:
            open_connection.close()

    if frame.empty:
        raise ValueError(
            f"no observations returned by {config.describe()} -- is wx_data populated?"
        )
    return frame
