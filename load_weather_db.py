#!/usr/bin/env python3
"""Load the daily observation CSV into weather_predict_db.

Reads the CSV fetch_weather.py writes and fills the schema bottom-up --
states, admin_div_1, cities, wmo_codes, then one wx_data row per city-day.
Connection details come from .env, the same ones db.py reads.

    python load_weather_db.py --create-schema     # first run: build the tables too
    python load_weather_db.py                     # load weather_daily.csv
    python load_weather_db.py --replace           # reload, discarding those city-days

Re-running is safe: a city-day already in wx_data is skipped rather than
duplicated, since the schema has no unique constraint to conflict on.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

import db
from config import CITY_COORDINATES

SCHEMA_FILE = "weather_predict_schema.sql"

# states.state_abbr wants an ISO 3-letter code; the geocoding cache only carries the 2-letter one.
ISO3 = {
    "US": "USA", "GB": "GBR", "JP": "JPN", "AU": "AUS", "BR": "BRA",
    "CA": "CAN", "DE": "DEU", "FR": "FRA", "IN": "IND", "MX": "MEX",
}

# WMO 4677 present-weather text, trimmed to the 40 characters wmo_codes.wmo_text allows.
WMO_TEXT = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snow fall", 73: "Moderate snow fall", 75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def city_places(path: str | Path = CITY_COORDINATES) -> dict[str, dict]:
    """Country and admin division per city, from the geocoding cache, keyed by name."""
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(
            f"{file} not found -- run `python fetch_weather.py` first; the schema needs "
            "each city's country and admin division, which the CSV does not carry"
        )
    return {entry["name"]: entry for entry in json.loads(file.read_text(encoding="utf-8")).values()}


def _one(cursor, query: str, params: tuple):
    cursor.execute(query, params)
    row = cursor.fetchone()
    return None if row is None else row[0]


def _find_or_add(cursor, schema: str, table: str, key: str, key_value: str,
                 columns: tuple[str, ...], values: tuple, id_column: str) -> int:
    """Reference rows are keyed by name -- the schema has no unique constraint to upsert on."""
    found = _one(cursor, f"SELECT {id_column} FROM {schema}.{table} WHERE {key} = %s",
                 (key_value,))
    if found is not None:
        return int(found)
    placeholders = ", ".join(["%s"] * len(values))
    return int(_one(cursor,
                    f"INSERT INTO {schema}.{table} ({', '.join(columns)}) "
                    f"VALUES ({placeholders}) RETURNING {id_column}", values))


def load_reference(cursor, schema: str, cities: list[str], places: dict[str, dict],
                   codes: list[int], say=print) -> dict[str, int]:
    """Fill states, admin_div_1, cities and wmo_codes; return city name -> city_id."""
    city_ids: dict[str, int] = {}
    for city in cities:
        place = places.get(city)
        if place is None:
            raise ValueError(
                f"no geocoding entry for {city!r} in {CITY_COORDINATES} -- "
                "refresh it with `python fetch_weather.py --refresh`"
            )
        country, code = place.get("country", ""), place.get("country_code", "")
        abbr = ISO3.get(code)
        if not abbr:
            raise ValueError(
                f"no ISO 3-letter code known for {country!r} ({code!r}) -- "
                f"add it to ISO3 in {Path(__file__).name}"
            )

        state_id = _find_or_add(cursor, schema, "states", "state_abbr", abbr,
                               ("state_name_long", "state_name_short", "state_abbr"),
                               (country[:60], country[:40], abbr), "state_id")
        # Cities without an admin1 (city-states) stand in their own division rather than violating div1_name NOT NULL.
        div1_name = (place.get("admin1") or country or city)[:60]
        div1_id = _find_or_add(cursor, schema, "admin_div_1", "div1_name", div1_name,
                              ("div1_name", "state_id"), (div1_name, state_id), "div1_id")
        city_ids[city] = _find_or_add(cursor, schema, "cities", "city_name", city,
                                     ("city_name", "div1_id"), (city[:60], div1_id),
                                     "city_id")

    # wmo_codes has a primary key, so this one can upsert.
    known = [(str(code), WMO_TEXT[code][:40]) for code in codes if code in WMO_TEXT]
    unknown = sorted({code for code in codes if code not in WMO_TEXT})
    if unknown:
        raise ValueError(f"no wmo_text for code(s) {unknown} -- add them to WMO_TEXT")
    cursor.executemany(
        f"INSERT INTO {schema}.wmo_codes (wmo_code, wmo_text) VALUES (%s, %s) "
        "ON CONFLICT (wmo_code) DO NOTHING", known)

    say(f"Reference data: {len(city_ids)} cities, {len(known)} WMO codes")
    return city_ids


def observation_rows(raw: pd.DataFrame, city_ids: dict[str, int]) -> list[tuple]:
    # dtg is midnight UTC so a day survives the round trip: db.py groups by (dtg AT TIME ZONE DB_TIMEZONE)::date, and DB_TIMEZONE defaults to UTC.
    return [
        (
            city_ids[row.city],
            f"{pd.Timestamp(row.date).date()} 00:00:00+00",
            None if pd.isna(row.temp_mean_c) else round(float(row.temp_mean_c), 1),
            None if pd.isna(row.precip_mm) else int(round(float(row.precip_mm))),
            None if pd.isna(row.wmo_code) else str(int(row.wmo_code)),
        )
        for row in raw.itertuples(index=False)
    ]


def load_observations(cursor, schema: str, rows: list[tuple], replace: bool,
                      say=print) -> int:
    if replace:
        cursor.execute(
            f"DELETE FROM {schema}.wx_data WHERE city_id = ANY(%s) "
            "AND dtg BETWEEN %s AND %s",
            (sorted({row[0] for row in rows}),
             min(row[1] for row in rows), max(row[1] for row in rows)))
        say(f"Replacing: deleted {cursor.rowcount:,} existing row(s)")

    # Staged first so the insert can skip city-days already present -- wx_data has no unique key to conflict on.
    cursor.execute("CREATE TEMP TABLE wx_stage (city_id int, dtg timestamptz, "
                   "temp_c decimal(4,1), precip_mm int, wmo_code char(2)) ON COMMIT DROP")
    cursor.executemany("INSERT INTO wx_stage VALUES (%s, %s, %s, %s, %s)", rows)
    cursor.execute(
        f"INSERT INTO {schema}.wx_data (city_id, dtg, temp_C, precip_mm, wmo_code) "
        "SELECT s.city_id, s.dtg, s.temp_c, s.precip_mm, s.wmo_code FROM wx_stage s "
        f"WHERE NOT EXISTS (SELECT 1 FROM {schema}.wx_data w "
        "                   WHERE w.city_id = s.city_id AND w.dtg = s.dtg)")
    return cursor.rowcount


def load(raw, places: dict[str, dict], config: db.DbConfig | None = None, *,
         replace: bool = False, schema_first: bool = False,
         say=print) -> tuple[int, int]:
    """Insert observations into wx_data; returns (inserted, skipped).

    raw is anything pandas can turn into the CSV's columns -- a DataFrame from
    load_raw(), or the row dicts fetch_weather.py already has in memory.
    """
    # Imported here: fetch_weather.py only needs this when asked to write to the database, and xg pulls in xgboost.
    from xg import normalize_raw

    config = db.DbConfig.from_env() if config is None else config
    frame = normalize_raw(raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw),
                          "the fetched observations")

    with db.open_connection(config) as connection:
        with connection.cursor() as cursor:
            if schema_first:
                create_schema(cursor, say=say)

            cities = sorted(frame["city"].unique())
            codes = sorted({int(code) for code in frame["wmo_code"].dropna().unique()})
            city_ids = load_reference(cursor, config.schema, cities, places, codes, say)

            rows = observation_rows(frame, city_ids)
            inserted = load_observations(cursor, config.schema, rows, replace, say)
        connection.commit()

    # precip_mm is an integer column, so anything under half a millimetre lands as 0 and reads as a dry day downstream.
    rounded = int(((frame["precip_mm"] > 0) & (frame["precip_mm"] < 0.5)).sum())
    if rounded:
        say(f"Note: {rounded:,} row(s) with 0 < precip < 0.5 mm stored as 0 "
            "-- wx_data.precip_mm is an integer column")
    return inserted, len(rows) - inserted


def create_database(config: db.DbConfig, say=print) -> None:
    """CREATE DATABASE, connecting to the maintenance database to do it."""
    import psycopg2
    from psycopg2 import sql

    maintenance = db.DbConfig(**{**vars(config), "name": "postgres"})
    connection = db.connect(maintenance)
    try:
        connection.autocommit = True  # CREATE DATABASE cannot run in a transaction
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (config.name,))
            if cursor.fetchone():
                say(f"Database {config.name} already exists")
                return
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(
                sql.Identifier(config.name)))
            say(f"Created database {config.name}")
    except psycopg2.Error as err:
        raise RuntimeError(f"could not create database {config.name}: {err}") from err
    finally:
        connection.close()


def create_schema(cursor, path: str | Path = SCHEMA_FILE, say=print) -> None:
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"{file} not found")
    cursor.execute(file.read_text(encoding="utf-8"))
    say(f"Applied {file}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="weather_daily.csv",
                        help="observation CSV to load (default: weather_daily.csv)")
    parser.add_argument("--create-database", action="store_true",
                        help="CREATE DATABASE DB_NAME first if it does not exist")
    parser.add_argument("--create-schema", action="store_true",
                        help=f"run {SCHEMA_FILE} before loading")
    parser.add_argument("--replace", action="store_true",
                        help="delete the CSV's city-days from wx_data before inserting")
    parser.add_argument("--quiet", action="store_true", help="only report the totals")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    def say(message: str = "") -> None:
        if not args.quiet:
            print(message)

    config = db.DbConfig.from_env()
    try:
        from xg import load_raw

        raw = load_raw(args.data)
        places = city_places()
        say(f"Read {len(raw):,} rows from {args.data} "
            f"({raw['date'].min().date()} -> {raw['date'].max().date()})")

        if args.create_database:
            create_database(config, say)

        say(f"Loading into {config.describe()}")
        inserted, skipped = load(raw, places, config, replace=args.replace,
                                 schema_first=args.create_schema, say=say)
        print(f"Inserted {inserted:,} wx_data row(s)"
              + (f", skipped {skipped:,} already present" if skipped else ""))
    except (FileNotFoundError, ValueError, RuntimeError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
