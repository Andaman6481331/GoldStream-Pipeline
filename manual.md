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
- **Transformation**: Normalizes disparate fields into a single `UnifiedTick` schema (Source, Timestamp, Symbol, Bid, Ask, Volume).

### Step 3: Feature Engineering (Gold Layer)
This is where the raw ticks become "intelligent."
- **Resampling**: Ticks are grouped into **5-minute OHLC candles** (Open, High, Low, Close).
- **Indicator Calculation**: Calculates **RSI (Momentum)** and **ATR (Volatility)** using the `ta` library.
- **Liquidity Detection**: Identifies potential support/resistance "Gaps" (Spread-aware) and "Round Number" levels.
- **Merge**: These 5-min features are mapped back to every single tick using a `merge_asof` logic.

### Step 4: Storage (DuckDB Feature Store)
Processed "Gold" data is stored in **DuckDB**, a high-performance local columnar database.
- **Why?**: DuckDB allows you to run complex SQL queries over millions of ticks in milliseconds, making it ideal for quantitative research.

### Step 5: Backtesting (Application Layer)
The library provides an **Event-Based Backtest Engine**.
- **Process**: The engine "replays" the ticks from DuckDB one by one.
- **Callbacks**: Your strategy receives a `TickEvent` (Price + Indicators) and returns an `Action` (Buy/Sell/Close).
- **Execution**: The engine handles trailing stops, spread-cost calculation, and liquidity gap force-closes.

---

## 🧠 Internal Logic Summary

- **Medallion Architecture**: Data flows from Raw (Bronze) → Validated (Silver) → Enriched (Gold).
- **Asynchronicity**: Uses `asyncio` for high-speed concurrent downloads and database I/O to handle thousands of ticks per second.
- **Micro-Normalization**: Ensures that whether data comes from a live MT5 stream or a 10-year-old file, it looks identical to your trading strategy.

---

## 📥 Inputs & 📤 Outputs

| Category | Input / Source | Output / Result |
| :--- | :--- | :--- |
| **Ingestion** | MT5 Python API & Dukascopy CDN (.bi5) | Partitioned Parquet (Year/Month) |
| **Validation** | Pydantic Schemas | Cleaned, Unified Dataframes |
| **Analytics** | Tick Mid-prices | RSI(14), ATR(14), Swing Highs/Lows |
| **Storage** | Local Memory / Parquet Files | `data/gold/goldstream.duckdb` (SQL-Ready) |
| **Backtest** | User Strategy Function | PnL Logs, Win-Rate, Max Drawdown |

---

## Visualize Data
```bash
duckdb -ui data/gold/goldstream.duckdb
```

---

## ⚡ Quick Commands
- **Ingest History**: `python ingest_history.py --start 2024-01-01 --end 2024-01-31`
- **Run Simulator**: `python example_backtest.py`
- **Check DB**: `SELECT count(*) FROM tick_features`
- **Check Parquet**: `python -c "import duckdb; print(duckdb.query(\"SELECT count(*) FROM 'data/bronze/XAUUSD/year=2023/month=2/ticks.parquet'\").fetchone())"`
