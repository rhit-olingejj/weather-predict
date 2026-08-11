import pytest
import pandas as pd
import numpy as np
from get_weather_data_from_city_date import get_info_from_city_date

def test_get_weather_data_from_city_date_no_city():
    city = None
    dtg = "2021-07-31 00:00:00-07"
    no_city = get_info_from_city_date(city, dtg)

    assert no_city == "no city given"

def test_get_weather_data_from_city_date_no_date():
    city = "New York"
    dtg = None
    no_dtg = get_info_from_city_date(city, dtg)

    assert no_dtg == "no dtg given"