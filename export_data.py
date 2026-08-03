"""
author: Daniel Schuster
script to export database information in JSON or CSV format

uses database information and credentials from environment variables
to connect.  copy .env.example to .env and fill out required info

command-line options:
  -h, --help            show this help message and exit
  -f, --format {csv,json}
                        Output format (csv or json)
  -o, --output OUTPUT   Output filename (e.g., export.csv)
  -d, --days DAYS       How many days back to include data

"""
import argparse
import psycopg2
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

# constants and configuration
load_dotenv()

# safely load env variables and fallback to empty strings if missing
DB_NAME = os.getenv("DB_NAME", "").strip()
DB_USER = os.getenv("DB_USER", "").strip()
DB_PASSWORD = os.getenv("DB_PASSWORD", "").strip()
DB_HOST = os.getenv("DB_HOST", "").strip()
DB_PORT = os.getenv("DB_PORT", "").strip()

SCHEMA_NAME = "weather_predict_db"


def get_cutoff_date_str(days):
    """
    calculates the cutoff date string based on days argument
    """
    cutoff_date = datetime.now() - timedelta(days=days)
    return cutoff_date.strftime('%Y-%m-%d %H:%M:%S')


def build_query(days):
    """
    generates the SQL query with the correct cutoff date
    """
    cutoff_date_str = get_cutoff_date_str(days)
    return f"""
    SELECT 
        w.dtg,
        c.city_name,
        a.div1_name AS admin_region,
        s.state_name_short AS state_country,
        s.state_abbr,
        w.temp_C,
        w.humidity,
        w.precip_chance,
        w.precip_mm,
        wc.wmo_text AS weather_condition
    FROM WX_DATA w
    JOIN CITIES c ON w.city_id = c.city_id
    JOIN ADMIN_DIV_1 a ON c.div1_id = a.div1_id
    JOIN STATES s ON a.state_id = s.state_id
    LEFT JOIN WMO_CODES wc ON w.wmo_code = wc.wmo_code
    WHERE w.dtg >= '{cutoff_date_str}'
    ORDER BY w.dtg DESC;
    """

def save_dataframe(df, file_format, output_file):
    """
    saves the dataframe to the requested file format (csv or json)
    """
    if file_format == 'csv':
        df.to_csv(output_file, index=False)
    elif file_format == 'json':
        df.to_json(output_file, orient='records', date_format='iso', indent=4)


def main():
    parser = argparse.ArgumentParser(description="Export weather data to CSV or JSON")
    
    parser.add_argument('-f', '--format', choices=['csv', 'json'], required=True, 
                        help="Output format (csv or json)")
    parser.add_argument('-o', '--output', required=True, 
                        help="Output filename (ex: export.csv)")
    parser.add_argument('-d', '--days', type=int, required=True, 
                        help="How many days back to include data")

    args = parser.parse_args()
    print(f"Extracting data from {get_cutoff_date_str(args.days)} to present day")

    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT,
            options=f"-c search_path={SCHEMA_NAME},public" #pass schema name in connection
        )

        query = build_query(args.days)
        df = pd.read_sql_query(query, conn)

        if df.empty:
            print("No data found for the specified timeframe.")
            return

        save_dataframe(df, args.format, args.output)
        print(f"Exported {len(df)} records to {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        # close connection if it was successfully created
        if 'conn' in locals() and conn:
            conn.close()


if __name__ == "__main__":
    main()
