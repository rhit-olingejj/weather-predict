"""
Author: Daniel Schuster

flags:
    -i, --input <input_csv_filepath>: required.  sets the input CSV file to insert data from
    --init: optional, initializes the database schema.  should only be run once
"""

import os
import csv
import argparse
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# constants and configuration
load_dotenv()
DB_NAME = os.getenv("DB_NAME").strip()
DB_USER = os.getenv("DB_USER").strip()
DB_PASSWORD = os.getenv("DB_PASSWORD").strip()
DB_HOST = os.getenv("DB_HOST").strip()
DB_PORT = os.getenv("DB_PORT").strip()

SCHEMA_FILE = "weather_predict_schema.sql"
SCHEMA_NAME = "weather_predict_db"

WMO_DESCRIPTIONS = {
    "0": "Clear sky", "1": "Mainly clear", "2": "Partly cloudy", "3": "Overcast",
    "45": "Fog", "48": "Depositing rime fog", "51": "Light drizzle",
    "53": "Moderate drizzle", "55": "Dense drizzle", "61": "Slight rain",
    "63": "Moderate rain", "65": "Heavy rain", "71": "Slight snow fall",
    "73": "Moderate snow fall", "75": "Heavy snow fall", "80": "Slight rain showers",
    "81": "Moderate rain showers", "82": "Violent rain showers", "95": "Thunderstorm"
}


def initialize_schema(conn, schema_file):
    """
    Performs initialization of the schema from schema_file.  
    This function should only run when the --init flag is used.
    """
    print(f"Applying schema from {schema_file}...")
    if not os.path.exists(schema_file):
        raise FileNotFoundError(f"Schema file not found at {schema_file}")
        
    with conn.cursor() as cur:
        with open(schema_file, 'r') as file:
            cur.execute(file.read())
    conn.commit()
    print("Schema applied")


def get_or_create(cur, table, id_col, check_col, check_val, insert_cols, insert_vals):
    """
    Helper function to fetch an existing ID, or create it if it doesn't exist.
    """
    cur.execute(f"SELECT {id_col} FROM {table} WHERE {check_col} = %s", (check_val,))
    result = cur.fetchone()
    
    if result:
        return result[0]
    
    # if ID doesn't exist, insert it
    placeholders = ", ".join(["%s"] * len(insert_vals))
    cur.execute(f"""
        INSERT INTO {table} ({insert_cols}) 
        VALUES ({placeholders}) RETURNING {id_col};""",
        insert_vals)
    return cur.fetchone()[0]


def process_csv(conn, csv_filepath):
    print(f"Reading CSV: {csv_filepath}...")
    city_cache = {}   
    wmo_cache = set() 
    wx_data_batch = []

    with open(csv_filepath, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        with conn.cursor() as cur:
            cur.execute(f"SET search_path TO {SCHEMA_NAME}, public;")

            # get or create Default State
            state_id_cache = get_or_create(
                cur, "STATES", "state_id", "state_abbr", "UNK",
                "state_name_long, state_name_short, state_abbr",
                ("Unknown State", "Unknown State", "UNK")
            )

            # get or create Default Admin Division
            div1_id_cache = get_or_create(
                cur, "ADMIN_DIV_1", "div1_id", "div1_abbr", "N/A",
                "div1_name, div1_abbr, state_id",
                ("Unknown Region", "N/A", state_id_cache)
            )

            for row in reader:
                city_name = row['city'].strip()

                # get or create CITIES
                if city_name not in city_cache:
                    city_id = get_or_create(
                        cur, "CITIES", "city_id", "city_name", city_name,
                        "city_name, div1_id",
                        (city_name, div1_id_cache)
                    )
                    city_cache[city_name] = city_id

                city_id = city_cache[city_name]

                # process WMO_CODES
                raw_wmo = str(row['wmo_code']).strip()
                if raw_wmo not in wmo_cache:
                    wmo_text = WMO_DESCRIPTIONS.get(raw_wmo, "Unknown condition")
                    cur.execute("""
                        INSERT INTO WMO_CODES (wmo_code, wmo_text)
                        VALUES (%s, %s) ON CONFLICT (wmo_code) DO NOTHING;
                    """, (raw_wmo, wmo_text))
                    wmo_cache.add(raw_wmo)

                # format row for WX_DATA table (uses zeros for humidity and precipitation chance currently)
                wx_data_batch.append((
                    city_id, row['date'], row['temp_mean_c'], 
                    0.0, 0.0, int(float(row['precip_mm'])), raw_wmo
                ))

            # insert weather records
            print(f"Parsed {len(wx_data_batch)} rows. Inserting into WX_DATA...")
            
            insert_query = """
                INSERT INTO WX_DATA (city_id, dtg, temp_C, humidity, precip_chance, precip_mm, wmo_code)
                VALUES %s;
                """
            psycopg2.extras.execute_values(cur, insert_query, wx_data_batch)

    conn.commit()
    print(f"Data imported from {csv_filepath}")


def main():
    parser = argparse.ArgumentParser(description="Import weather from CSV into PostgreSQL.")
    parser.add_argument('-i', '--input', required=True, help="Path to CSV input file")
    parser.add_argument('--init', action='store_true', help="Initialize database schema before importing")
    args = parser.parse_args()

    if not DB_PASSWORD:
        print("Error: DB_PASSWORD missing from .env file.")
        return

    try:
        conn = psycopg2.connect(
                host=DB_HOST,
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                port=DB_PORT,
                )
        
        # only run initialization if explicitly requested with --init flag
        if args.init:
            initialize_schema(conn, SCHEMA_FILE)
            
        process_csv(conn, args.input)

    except psycopg2.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()
            print("Database connection closed.")


if __name__ == "__main__":
    main()
