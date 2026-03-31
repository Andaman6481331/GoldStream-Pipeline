import duckdb
import pandas as pd
try:
    con = duckdb.connect('data/gold/goldstream.duckdb')
    df = con.execute("SELECT MIN(tick_time), MAX(tick_time), COUNT(*) FROM trade_decisions").df()
    print("Trade Decisions Stats:")
    print(df)
    
    sample = con.execute("SELECT * FROM trade_decisions LIMIT 3").df()
    print("\nSample Data:")
    print(sample)
    
    con.close()
except Exception as e:
    print(f"Error: {e}")
