import duckdb
import os

db_path = "data/gold/goldstream.duckdb"
if not os.path.exists(db_path):
    print(f"Database not found at {db_path}")
else:
    con = duckdb.connect(db_path)
    tables = con.execute("SHOW TABLES;").fetchall()
    print("Tables in DuckDB:")
    for t in tables:
        print(f" - {t[0]}")
        
    for table in ["candles_1m", "candles_15m", "candles_4h"]:
        if any(t[0] == table for t in tables):
            cols = con.execute(f"DESCRIBE {table};").fetchall()
            print(f"\nColumns in {table}:")
            for c in cols:
                print(f"  {c[0]} ({c[1]})")
        else:
            print(f"\nTable {table} NOT FOUND")
    con.close()
