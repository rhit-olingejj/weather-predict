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

# XGBoost training process

How `xg.py` turns `weather_daily.csv` into the three next-day models that
`predict_weather.py` serves.

## End to end

```mermaid
flowchart TD
    CSV[("weather_daily.csv<br/>9,135 daily rows · 5 cities · 2020-2024")]
    RAW["load_raw()<br/>parse dates · sort by city, date · drop duplicates"]
    FEAT["build_features()<br/>one supervised row per (city, day D)"]
    DROP["drop warm-up rows<br/>first 29 days per city lack a full 30-day window"]
    LABEL["LabelEncoder on y_condition<br/>Clear · Cloudy · Drizzle · Rain · Snow"]
    XY["X = 65 numeric features<br/>y = 3 target columns<br/>8,985 rows"]
    SPLIT["Train_validate_test_split()<br/>random 64 / 16 / 20"]

    TRAIN["train<br/>5,750 rows"]
    VAL["validate<br/>1,438 rows"]
    TEST["test<br/>1,797 rows"]

    SEARCH["parameter search<br/>10 candidates x 3 targets"]
    BEST["best model per target"]
    SCORE["score once on test"]
    BUNDLE[("models/xgb_weather.joblib<br/>boosters + feature schema + metrics")]
    SERVE["predict_weather.py"]

    CSV --> RAW --> FEAT --> DROP --> LABEL --> XY --> SPLIT
    SPLIT --> TRAIN
    SPLIT --> VAL
    SPLIT --> TEST
    TRAIN -->|fit| SEARCH
    VAL -->|early stopping + candidate ranking| SEARCH
    SEARCH --> BEST
    BEST --> SCORE
    TEST -->|never seen during selection| SCORE
    SCORE --> BUNDLE --> SERVE

    classDef data fill:#e8eef7,stroke:#4a6fa5,color:#1a2a3a
    classDef split fill:#eef3e8,stroke:#6a8f4a,color:#1a2a3a
    classDef model fill:#f7efe4,stroke:#b07d3a,color:#2a1f10
    class CSV,BUNDLE data
    class TRAIN,VAL,TEST split
    class SEARCH,BEST,SCORE,SERVE model
```

## Feature construction

Every feature describes day `D` or earlier; every label describes `D+1`. That
boundary is the whole point of the function — nothing that would be unknowable
at forecast time may enter the matrix.

```mermaid
flowchart LR
    OBS["observations for day D<br/>temp mean/max/min · precip_mm · wmo_code"]

    subgraph inputs ["features — day D and earlier"]
        TODAY["today<br/>raw values · temp_range_c · is_wet<br/>cond_today_* one-hots"]
        LAGS["lags<br/>1, 2, 3, 7 days back<br/>x 5 columns"]
        ROLL["trailing windows<br/>mean/std/sum over 3, 7, 14, 30 days"]
        DELTA["momentum<br/>day-over-day deltas<br/>anomaly vs 7d and 30d mean"]
        WHERE["place<br/>latitude · longitude · city_* one-hots"]
        WHEN["season of D+1<br/>doy sin/cos · month<br/>hemisphere-signed wave"]
    end

    subgraph labels ["labels — day D+1"]
        YT["y_temp_mean_c<br/>regression"]
        YP["y_is_wet<br/>precip_mm > 0.1"]
        YC["y_condition_code<br/>WMO bucket"]
    end

    OBS --> TODAY
    OBS --> LAGS
    OBS --> ROLL
    OBS --> DELTA
    OBS -.->|shifted forward one day, consecutive days only| YT
    OBS -.-> YP
    OBS -.-> YC

    classDef feat fill:#e8eef7,stroke:#4a6fa5,color:#1a2a3a
    classDef lab fill:#f7e8e8,stroke:#a54a4a,color:#3a1a1a
    class TODAY,LAGS,ROLL,DELTA,WHERE,WHEN feat
    class YT,YP,YC lab
```

Calendar dates themselves are excluded. A split on `date > 2023-06-01` carves up
the training calendar and cannot transfer to any future day, so seasonality is
supplied only through cyclical encodings that repeat every year.

## The parameter search

`search()` runs once per target. Each candidate is fitted on train, early-stopped
on validation, and ranked by a single validation metric. All three metrics are
lower-is-better.

```mermaid
flowchart TD
    GRID["PARAM_GRID — 10 candidates<br/>shallow · baseline · slow-steady · wide-sparse · heavy-reg<br/>deep · deep-damped · aggressive · short-budget · long-slow"]
    FIT["fit candidate<br/>tree budget from n_estimators (60 to 2500)<br/>early_stopping_rounds = 50"]
    STOP{"validation score<br/>improved in last 50 rounds,<br/>and budget left?"}
    MORE["add another tree"]
    DONE["stop · keep best_iteration<br/>flag if the budget bound"]
    EVAL["score on validation"]
    RANK["rank all candidates<br/>temperature: MAE<br/>precipitation: log loss<br/>condition: log loss"]
    WIN["winner — used as fitted,<br/>not refit"]
    TESTED["score once on test"]

    GRID -->|for each of 10| FIT --> STOP
    STOP -->|yes| MORE --> STOP
    STOP -->|no| DONE --> EVAL
    EVAL -->|next candidate| GRID
    EVAL --> RANK --> WIN --> TESTED

    classDef choice fill:#f7efe4,stroke:#b07d3a,color:#2a1f10
    class STOP choice
```

The winner is kept exactly as fitted rather than refit on train + validation:
refitting would discard the early-stopping set that chose its tree count.

## Three targets, one feature matrix

```mermaid
flowchart LR
    X["X — 65 features<br/>identical for all three"]
    R["XGBRegressor<br/>reg:squarederror<br/>eval: MAE"]
    B["XGBClassifier<br/>binary:logistic<br/>eval: log loss"]
    M["XGBClassifier<br/>multi:softprob<br/>eval: mlogloss"]
    OUT["structured forecast<br/>temperature_c<br/>precipitation probability<br/>condition + class distribution"]

    X --> R --> OUT
    X --> B --> OUT
    X --> M --> OUT
```

Sharing one matrix and one split means the three targets are trained and scored
on exactly the same rows, so their metrics are directly comparable. Each target
picks its own winning configuration independently.

## What gets reported

| stage | output |
| --- | --- |
| per candidate | selection metric, trees used vs. budget, wall-clock seconds |
| per target | full candidate table sorted best-first, winner marked |
| winner | validation score sampled across boosting rounds |
| winner | validation and test scores against a naive baseline |
| winner | top features by gain |

Baselines the models are measured against: persistence ("tomorrow looks like
today") for temperature, and the majority class for both classifiers.

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
