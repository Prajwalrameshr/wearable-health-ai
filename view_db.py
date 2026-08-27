import psycopg2
import pandas as pd

url = 'postgresql://neondb_owner:npg_NE3L0pAzPuWt@ep-tiny-star-aviqtotw-pooler.c-11.us-east-1.aws.neon.tech/neondb?sslmode=require'

try:
    print("\n=========================================================================================")
    print("                      NEON POSTGRESQL LIVE DATABASE VIEWER                               ")
    print("=========================================================================================\n")

    conn = psycopg2.connect(url)
    df = pd.read_sql_query('''
        SELECT 
            id, 
            device_user_id, 
            record_date, 
            steps, 
            distance_km, 
            calories, 
            heart_rate AS avg_hr, 
            oxygen_saturation AS spo2, 
            sleep_minutes, 
            predicted_state, 
            risk_score, 
            risk_level
        FROM health_logs 
        ORDER BY record_date DESC;
    ''', conn)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)

    print(df)
    print("\n=========================================================================================")
    print(f" Total Rows Returned: {len(df)}")
    print("=========================================================================================\n")

    conn.close()
except Exception as e:
    print("Error querying database:", e)
