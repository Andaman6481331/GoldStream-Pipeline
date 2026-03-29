# GoldStream ETL Pipeline — User Manual

## 📖 Introduction
GoldStream is a high-performance Data Engineering pipeline designed for **XAUUSD (Gold)** tick-level data. It transforms raw financial data from MetaTrader 5 and Dukascopy into a "production-ready" feature store for algorithmic trading and backtesting.

---

## 🛠️ Workflow (How it Works in 5 Steps)

### Step 1: Ingestion (Bronze Layer)
The system collects raw data from two sources:
1.  **Live Feed (MT5)**: Connects to MetaTrader 5 via `MetaTrader5` Python API to stream real-time ticks.
2.  **Historical Feed (Dukascopy)**: Fetches compressed `.bi5` binary files directly from the Dukascopy CDN.
- **Output**: Raw ticks saved as **Partitioned Parquet files** (`data/bronze/`) for high-speed storage.

### Step 2: Validation & Normalization (Silver Layer)
Raw data from different sources is often "dirty" or inconsistently formatted.
- **Logic**: Uses **Pydantic Models** to enforce strict data types (e.g., prices must be positive, timestamps must be valid).
- **Transformation**: Normalizes disparate fields into a single `UnifiedTick` schema.

### Step 3: Feature Engineering (Gold Layer)
This is where the raw ticks become "intelligent" following **SMC (Smart Money Concepts)** rules.
- **BOS/CHoCH Detection**: Automatically identifies market structure shifts.
-  **Fair Value Gaps (FVG)**: Identifies 3-candle price imbalances and tracks if they are "filled" or "mitigated".
- **Indicator Calculation**: RSI (Momentum) and ATR (Volatility).
- **Session Labelling**: Tags every tick as `london`, `new_york`, `asian`, or the `killzone` window.

### Step 4: Storage (DuckDB Feature Store)
Processed "Gold" data is stored in **DuckDB**, a high-performance local columnar database.
- **Why?**: DuckDB allows unified SQL access to both live and historical features with millisecond latency.
- **Persistence**: All decisions made by the strategy are also recorded in the `trade_decisions` table for audit.

### Step 5: Backtesting & Strategy (Application Layer)
The system runs the **Scout & Sniper** execution engine.
- **Scout Phase**: Monitors for a structure break (BOS) indicating a new trend.
- **Sniper Phase**: Once a trend is confirmed, it waits for price to retrace into a fresh FVG (Fair Value Gap) before executing.
- **Decision Persistence**: Every signal is saved to DuckDB with full context (RSI, ATR, FVG bounds).

---

## 🧠 Internal Logic Summary

- **Unified Logic**: Whether data is live or historical, it flows through the same `FeatureEngineer` and `Strategy` code.
- **Asynchronicity**: Uses `asyncio` for high-speed concurrent downloads and database ops.
- **Audit-Ready**: The DuckDB store acts as a single source of truth for both price data and trade signals.

---

## 📥 Inputs & 📤 Outputs

| Category | Input / Source | Output / Result |
| :--- | :--- | :--- |
| **Ingestion** | MT5 Python API & Dukascopy CDN | Partitioned Parquet (Year/Month) |
| **Validation** | Pydantic Schemas | Cleaned, Unified Dataframes |
| **Gold Analytics** | Tick mid-prices | BOS, CHoCH, FVG, RSI, ATR |
| **Storage** | Local Parquet / DuckDB | `goldstream.duckdb` (SQL-Ready) |
| **Trade Audit** | Scout & Sniper Strategy | `trade_decisions` table |

---

## ⚡ SQL Audit Commands
Use the DuckDB CLI to inspect your data:
```bash
# Open DuckDB UI
duckdb -ui data/gold/goldstream.duckdb

# Check how many structural breaks were detected
duckdb data/gold/goldstream.duckdb -s "SELECT session, count(*) FROM tick_features WHERE bos_detected = TRUE GROUP BY session"

# View the latest trade signals
duckdb data/gold/goldstream.duckdb -s "SELECT * FROM trade_decisions ORDER BY tick_time DESC LIMIT 10"
```

---

## ⚡ Quick Shell Commands
- **Remove Old Data DuckDB**: `rm data/gold/goldstream.duckdb` (bash)
- **Ingest & Audit History**: `python ingest_history.py --start 2024-03-01 --end 2024-03-07`
    - **Skip Download**: `python ingest_history.py --start 2024-03-01 --end 2024-03-07 --skip-download` - only if you already have the data in bronze(.parquet)
    - **Skip Silver & Download**: `python ingest_history.py --start 2024-03-01 --end 2024-03-07 --skip-silver --skip-download` - only if you already have the data in silver(.duckdb)
- **SMC Visual Audit**: `python src/bot/visualizer.py` (requires `pip install plotly`)
- **Run Live Pipeline**: `python main.py`
- **Check DB Ticks**: `python -c "import duckdb; print(duckdb.connect('data/gold/goldstream.duckdb', read_only=True).execute('SELECT COUNT(*) FROM tick_features').fetchone()[0])"`
