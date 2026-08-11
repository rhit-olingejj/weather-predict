#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"

# Daily variables requested from the archive API. Humidity and precipitation probability are only offered at hourly resolution, so they are not available here; the corresponding schema columns (humidity, precip_chance) are left null.
DAILY_VARS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "weather_code",
]


@dataclass(frozen=True)
class City:
    name: str
    latitude: float
    longitude: float
    timezone: str
    country: str = ""
    country_code: str = ""
    admin1: str = ""  # e.g. state / province -> maps to admin_div_1


# Cities to fetch, as (query, country_code) pairs. The country_code disambiguates common names (there are many "Londons") when geocoding.
CITY_QUERIES = [
    ("New York", "US"),
    ("London", "GB"),
    ("Tokyo", "JP"),
    ("Sydney", "AU"),
    ("São Paulo", "BR"),
]

DEFAULT_CACHE = "geocode_cache.json"

CSV_FIELDS = [
    "city",
    "latitude",
    "longitude",
    "date",
    "temp_mean_c",
    "temp_max_c",
    "temp_min_c",
    "precip_mm",
    "wmo_code",
]


def resolve_range(args: argparse.Namespace) -> tuple[date, date]:
    if args.end:
        end = datetime.strptime(args.end, "%Y-%m-%d").date()
    else:
        end = (datetime.now(timezone.utc) - timedelta(hours=24)).date()

    if args.start:
        start = datetime.strptime(args.start, "%Y-%m-%d").date()
    else:
        # Handle Feb 29 by falling back a day.
        try:
            start = end.replace(year=end.year - args.years)
        except ValueError:
            start = end.replace(year=end.year - args.years, day=end.day - 1)

    if start > end:
        sys.exit(f"start ({start}) must be on or before end ({end})")
    return start, end


def fetch_city(city: City, start: date, end: date, session: requests.Session,
               retries: int = 4) -> list[dict]:
    params = {
        "latitude": city.latitude,
        "longitude": city.longitude,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join(DAILY_VARS),
        "timezone": city.timezone,
    }

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = session.get(ARCHIVE_URL, params=params, timeout=60)
            if resp.status_code == 429:
                raise requests.HTTPError("429 rate limited", response=resp)
            resp.raise_for_status()
            return _rows_from_payload(city, resp.json())
        except (requests.RequestException, ValueError) as err:
            last_err = err
            wait = 2 ** attempt
            print(f"  ! {city.name}: attempt {attempt + 1} failed ({err}); "
                  f"retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)

    raise RuntimeError(f"failed to fetch {city.name} after {retries} attempts: {last_err}")


def _rows_from_payload(city: City, payload: dict) -> list[dict]:
    daily = payload.get("daily")
    if not daily or "time" not in daily:
        return []

    dates = daily["time"]
    rows = []
    for i, day in enumerate(dates):
        rows.append({
            "city": city.name,
            "latitude": city.latitude,
            "longitude": city.longitude,
            "date": day,
            "temp_mean_c": _at(daily, "temperature_2m_mean", i),
            "temp_max_c": _at(daily, "temperature_2m_max", i),
            "temp_min_c": _at(daily, "temperature_2m_min", i),
            "precip_mm": _at(daily, "precipitation_sum", i),
            "wmo_code": _at(daily, "weather_code", i),
        })
    return rows


def _at(daily: dict, key: str, i: int):
    values = daily.get(key)
    return values[i] if values and i < len(values) else None


def geocode(query: str, country_code: str, session: requests.Session,
            retries: int = 4) -> City:
    params = {"name": query, "count": 10, "language": "en", "format": "json"}

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = session.get(GEOCODE_URL, params=params, timeout=60)
            if resp.status_code == 429:
                raise requests.HTTPError("429 rate limited", response=resp)
            resp.raise_for_status()
            results = resp.json().get("results") or []
            if not results:
                raise RuntimeError(f"no geocoding match for {query!r}")

            # Prefer a hit in the requested country; the API already ranks by population, so the first match is otherwise the best.
            match = next(
                (r for r in results
                 if not country_code or r.get("country_code") == country_code),
                results[0],
            )
            return City(
                name=match.get("name", query),
                latitude=match["latitude"],
                longitude=match["longitude"],
                timezone=match.get("timezone", "UTC"),
                country=match.get("country", ""),
                country_code=match.get("country_code", ""),
                admin1=match.get("admin1", ""),
            )
        except (requests.RequestException, ValueError) as err:
            last_err = err
            wait = 2 ** attempt
            print(f"  ! geocode {query}: attempt {attempt + 1} failed ({err}); "
                  f"retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)

    raise RuntimeError(f"failed to geocode {query!r} after {retries} attempts: {last_err}")


def resolve_cities(session: requests.Session, cache_path: Path,
                   refresh: bool) -> list[City]:
    cache: dict[str, dict] = {}
    if cache_path.exists() and not refresh:
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as err:
            print(f"  ! ignoring unreadable cache {cache_path} ({err})", file=sys.stderr)

    cities: list[City] = []
    dirty = False
    for query, country_code in CITY_QUERIES:
        key = f"{query}|{country_code}"
        if key in cache and not refresh:
            cities.append(City(**cache[key]))
            print(f"- {query}: cached")
            continue
        try:
            city = geocode(query, country_code, session)
        except RuntimeError as err:
            if key in cache:
                print(f"  ! {err}; falling back to cache", file=sys.stderr)
                cities.append(City(**cache[key]))
                continue
            raise
        cache[key] = asdict(city)
        cities.append(city)
        dirty = True
        print(f"- {query}: geocoded -> ({city.latitude}, {city.longitude}) "
              f"{city.timezone}")

    if dirty:
        cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False),
                              encoding="utf-8")
        print(f"Cached {len(cache)} geocode result(s) to {cache_path}")
    return cities


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, default=5,
                        help="Years of history to pull when --start is omitted (default: 5)")
    parser.add_argument("--start", help="Start date YYYY-MM-DD (overrides --years)")
    parser.add_argument("--end", help="End date YYYY-MM-DD (default: 24h ago)")
    parser.add_argument("--out", default="weather_daily.csv",
                        help="Output CSV path (default: weather_daily.csv)")
    parser.add_argument("--cache", default=DEFAULT_CACHE,
                        help=f"Geocoding cache path (default: {DEFAULT_CACHE})")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-geocode and overwrite the cache")
    parser.add_argument("--to-db", action="store_true",
                        help="Insert the fetched rows into weather_predict_db (see .env)")
    parser.add_argument("--create-database", action="store_true",
                        help="With --to-db: CREATE DATABASE DB_NAME if it is missing")
    parser.add_argument("--create-schema", action="store_true",
                        help="With --to-db: run weather_predict_schema.sql first")
    parser.add_argument("--replace", action="store_true",
                        help="With --to-db: delete these city-days before inserting")
    parser.add_argument("--no-csv", action="store_true",
                        help="With --to-db: skip writing the CSV")
    args = parser.parse_args()

    start, end = resolve_range(args)

    all_rows: list[dict] = []
    with requests.Session() as session:
        session.headers.update({"User-Agent": "weather-predict/1.0"})

        print(f"Resolving {len(CITY_QUERIES)} city centers")
        cities = resolve_cities(session, Path(args.cache), args.refresh)

        print(f"Fetching daily weather {start} -> {end} for {len(cities)} cities")
        for city in cities:
            print(f"- {city.name} ({city.latitude}, {city.longitude})")
            rows = fetch_city(city, start, end, session)
            print(f"    {len(rows)} days")
            all_rows.extend(rows)

    if not (args.to_db and args.no_csv):
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Wrote {len(all_rows)} rows to {args.out}")

    if args.to_db:
        return insert(all_rows, cities, args)
    return 0


def insert(rows: list[dict], cities: list[City], args: argparse.Namespace) -> int:
    """Hand the fetched rows to load_weather_db, no CSV round trip in between."""
    # Imported here so a plain CSV fetch needs neither psycopg2 nor pandas.
    import db
    import load_weather_db

    # The schema wants each city's country and admin division, which geocoding already resolved -- no need to re-read the cache file.
    places = {
        city.name: {"country": city.country, "country_code": city.country_code,
                    "admin1": city.admin1}
        for city in cities
    }

    config = db.DbConfig.from_env()
    try:
        if args.create_database:
            load_weather_db.create_database(config)
        print(f"Loading into {config.describe()}")
        inserted, skipped = load_weather_db.load(
            rows, places, config, replace=args.replace,
            schema_first=args.create_schema)
    except (FileNotFoundError, ValueError, RuntimeError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    print(f"Inserted {inserted:,} wx_data row(s)"
          + (f", skipped {skipped:,} already present" if skipped else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
