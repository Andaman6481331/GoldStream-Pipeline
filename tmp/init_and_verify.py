from src.gold.duckdb_store import DuckDBStore
import os

db_path = "data/gold/goldstream.duckdb"
store = DuckDBStore(db_path)
store.init_schema()

import duckdb
con = duckdb.connect(db_path)
tables = con.execute("SHOW TABLES;").fetchall()
print("Tables in DuckDB after init_schema:")
for t in tables:
    print(f" - {t[0]}")
    
for table in ["candles_1m", "candles_15m", "candles_4h"]:
    if any(t[0] == table for t in tables):
        cols = con.execute(f"DESCRIBE {table};").fetchall()
        print(f"\nColumns in {table}:")
        for c in cols:
            print(f"  {c[0]} ({c[1]})")
con.close()
store.close()
