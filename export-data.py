"""
author: Daniel Schuster
script to export database information in JSON or CSV format

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
from datetime import datetime, timedelta

def main():
    parser = argparse.ArgumentParser(description="Export weather data to CSV or JSON")
    
    parser.add_argument('-f', '--format', 
                        choices=['csv', 'json'], 
                        required=True, 
                        help="Output format (csv or json)")
                        
    parser.add_argument('-o', '--output', 
                        required=True, 
                        help="Output filename (ex: export.csv)")
                        
    parser.add_argument('-d', '--days', 
                        type=int, 
                        required=True, 
                        help="How many days back to include data")

    args = parser.parse_args()

    # calculate the time window based on '--days' argument
    cutoff_date = datetime.now() - timedelta(days=args.days)
    
    # format the date to match SQL standard DATETIME (YYYY-MM-DD HH:MM:SS)
    cutoff_date_str = cutoff_date.strftime('%Y-%m-%d %H:%M:%S')
    print(f"Extracting data from {cutoff_date_str} to present...")

    # connect to your database
    # NOTE: replace this with your actual database connection!
    conn = psycopg2.connect("dbname=test user=postgres password=secret")

    # define SQL Query (Joining all tables to create a flat export)
    query = f"""
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

    try:
        # run query and load directly into dataframe 
        df = pd.read_sql_query(query, conn)

        if df.empty:
            print("No data found for the specified timeframe.")
            return

        # export as csv or json based on argument 
        if args.format == 'csv':
            df.to_csv(args.output, index=False)
        elif args.format == 'json':
            df.to_json(args.output, orient='records', date_format='iso', indent=4)

        print(f"Exported {len(df)} records to {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()
