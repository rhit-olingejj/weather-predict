# weather-predict

Next-day weather forecasting with XGBoost, trained on five years of daily
observations for New York, London, Tokyo, Sydney and São Paulo.

| file | role |
| --- | --- |
| `fetch_weather.py` | pull daily history from the Open-Meteo archive into `weather_daily.csv` |
| `weather_predict_schema.sql` | the PostgreSQL schema the same observations live in |
| `db.py` | roll `wx_data` up to city-days and hand them to training |
| `xg.py` | build features, search parameters, train and save the three models |
| `predict_weather.py` | load the bundle and serve a structured forecast |
| `train_validate_test_split.py` | the 64/16/20 split helper |
| `config.py` | paths, thresholds and `.env` loading, shared by training and serving |
| `conftest.py` | synthetic fixtures — no test touches the real CSV or the database |

```console
$ pip install -r requirements.txt
$ cp .env.example .env                    # then fill in the database credentials
$ python xg.py                            # search + train, ~2 min
$ python predict_weather.py --city "New York" --date 2024-12-31
```

## Where the observations come from

`DATA_SOURCE` in `.env` picks the default (`db` or `csv`) and `--source`
overrides it per run, so both entry points read either place:

```console
$ python xg.py --source db                       # PostgreSQL, per .env
$ python fetch_weather.py                        # writes weather_daily.csv
$ python xg.py --source csv --data weather_daily.csv
```

`.env` is gitignored; `.env.example` documents every setting
(`DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_SCHEMA`,
`DB_SSLMODE`, or a single `DATABASE_URL`). Real environment variables win over
the file, so CI and containers can inject credentials instead. A trained bundle
records which source it came from and serving follows it unless told otherwise.

`wx_data` holds one row per observation timestamp while training wants one row
per city-day, so `db.py` aggregates: mean/max/min `temp_C`, summed `precip_mm`,
and the day's highest WMO code, with the day boundary drawn in `DB_TIMEZONE`.
Hourly rows give a true daily range; a database holding one row per day leaves
`temp_max_c == temp_min_c == temp_mean_c` and `temp_range_c` at zero.

## Tests

```console
$ python -m pytest -q
```

130 tests, ~8 seconds. Everything runs on synthetic observations built in
`conftest.py` — `weather_daily.csv` is gitignored and comes from a network
call, so no test may depend on it. The end-to-end fixture trains the real
pipeline on that synthetic data with a two-candidate grid, pinned to
`--source csv` so a filled-in `.env` can never point the suite at a real
database; `test_db.py` drives the database path through a fake connection.

The load-bearing ones guard the D / D+1 boundary: that every `y_*` column
equals the next day's observation, that no `y_*` column reaches
`feature_names()`, that a gap in the record suppresses labels rather than
silently forecasting two days ahead, and that the bundle `xg.py` writes still
splats cleanly into `WeatherPredictor.__init__`.
