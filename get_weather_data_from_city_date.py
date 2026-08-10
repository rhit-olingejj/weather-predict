# I could not test the happy path because I would need to connect to the actual database
import argparse
import psycopg2
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

DB_NAME = os.getenv("DB_NAME", "").strip()
DB_USER = os.getenv("DB_USER", "").strip()
DB_PASSWORD = os.getenv("DB_PASSWORD", "").strip()
DB_HOST = os.getenv("DB_HOST", "").strip()
DB_PORT = os.getenv("DB_PORT", "").strip()

SCHEMA_NAME = "weather_predict_db"

def query_get_info_from_city_date(city, dtg):
    return f"""
    SELECT 
        w.temp_C,
        w.humidity,
        w.precip_chance,
        w.precip_mm,
    FROM WX_DATA w
    JOIN CITIES c ON w.city_id = c.city_id
    WHERE w.dtg = '{dtg}' AND c.city_name = "{city}"
    """

def read_database(query): # query as a string
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT,
            options=f"-c search_path={SCHEMA_NAME},public" #pass schema name in connection
        )

        info = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Error: {e}")


def get_info_from_city_date(city, dtg) -> str:
    if (city == None):
        return "no city given"

    if (dtg == None):
        return "no dtg given"

    query = query_get_info_from_city_date(city, dtg)

    info = read_database(query)

    return info