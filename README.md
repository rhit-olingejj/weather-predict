# weather-predict

Next-day weather forecasting with XGBoost, trained on five years of daily
observations for New York, London, Tokyo, Sydney and São Paulo.

| file | role |
| --- | --- |
| `fetch_weather.py` | pull daily history from the Open-Meteo archive into `weather_daily.csv` |
| `xg.py` | build features, search parameters, train and save the three models |
| `predict_weather.py` | load the bundle and serve a structured forecast |
| `train_validate_test_split.py` | the 64/16/20 split helper |
| `config.py` | paths and thresholds shared by training and serving |
| `conftest.py` | synthetic fixtures — no test touches the real CSV |

```console
$ pip install -r requirements.txt
$ python fetch_weather.py                 # writes weather_daily.csv
$ python xg.py                            # search + train, ~2 min
$ python predict_weather.py --city "New York" --date 2024-12-31
```

## Tests

```console
$ python -m pytest -q
```

104 tests, ~8 seconds. Everything runs on synthetic observations built in
`conftest.py` — `weather_daily.csv` is gitignored and comes from a network
call, so no test may depend on it. The end-to-end fixture trains the real
pipeline on that synthetic data with a two-candidate grid.

The load-bearing ones guard the D / D+1 boundary: that every `y_*` column
equals the next day's observation, that no `y_*` column reaches
`feature_names()`, that a gap in the record suppresses labels rather than
silently forecasting two days ahead, and that the bundle `xg.py` writes still
splats cleanly into `WeatherPredictor.__init__`.
