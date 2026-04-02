import duckdb

conn = duckdb.connect('data/gold/goldstream.duckdb', read_only=True)

row = conn.execute("""
 DESCRIBE tick_features
""").fetchone()

labels = [
    "total_rows",
    "bos_rows",
    "has_sweep_low",
    "has_sweep_high",
    "has_fvg",
    "has_4h_bias",
]

# for label, value in zip(labels, row):
#     print(f"  {label:<25} {value:>10,}")

print(row)

conn.close()