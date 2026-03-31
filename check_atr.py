import duckdb
import pandas as pd
try:
    con = duckdb.connect('data/gold/goldstream.duckdb')
    df = con.execute("SELECT COUNT(*) as total, COUNT(atr_15_15m) as count_atr, MIN(timestamp_utc) as min_ts FROM tick_features WHERE symbol = 'XAUUSD'").df()
    print(df)
    
    null_count = con.execute("SELECT COUNT(*) FROM tick_features WHERE atr_15_15m IS NULL").fetchone()[0]
    print(f"Null atr_15_15m count: {null_count}")
    
    # Check the first few rows with data
    sample = con.execute("SELECT timestamp_utc, atr_15_15m FROM tick_features WHERE atr_15_15m IS NOT NULL LIMIT 5").df()
    print("\nFirst 5 rows with ATR data:")
    print(sample)
    
    con.close()
except Exception as e:
    print(f"Error: {e}")
