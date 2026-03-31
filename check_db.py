import duckdb
import pandas as pd
try:
    con = duckdb.connect('data/gold/goldstream.duckdb')
    df = con.execute("SELECT MIN(timestamp_utc) as min_ts, MAX(timestamp_utc) as max_ts, COUNT(*) as count FROM tick_features WHERE symbol = 'XAUUSD'").df()
    print(df)
    con.execute("SELECT timestamp_utc FROM tick_features WHERE symbol = 'XAUUSD' LIMIT 5").df().to_csv('sample_ts.csv', index=False)
    con.close()
except Exception as e:
    print(f"Error: {e}")
