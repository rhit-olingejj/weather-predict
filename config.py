"""Defaults shared by the training script (xg.py) and the serving API
(predict_weather.py).

These live here rather than in either module so the two never disagree about
where the model bundle is written and read, and so a path can be changed in one
place. Anything specific to one side -- model hyperparameters, feature windows,
output formatting -- stays in the module that owns it.
"""
from __future__ import annotations

# Where the observation CSV written by fetch_weather.py lives.
DEFAULT_DATA = "weather_daily.csv"

# Where xg.py writes the trained bundle and predict_weather.py looks for it.
DEFAULT_MODEL_DIR = "models"
MODEL_FILENAME = "xgb_weather.joblib"

# A day counts as "wet" above the WMO trace threshold; below that, gauges are
# mostly reporting dew and condensation rather than real precipitation. Both
# sides need this: training turns it into the label, serving reports it.
PRECIP_THRESHOLD_MM = 0.1
