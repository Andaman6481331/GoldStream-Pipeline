### GoldStream-Pipeline
Unified Data Engineering & Algorithmic Trading Infrastructure for **XAUUSD (Gold)**. 

This project implements a professional **Medallion Architecture** (Bronze, Silver, Gold) to transform raw, high-frequency tick data into a production-ready feature store for institutional-grade strategies like **Scout & Sniper**.

## 🎯 Project Objective
The goal is to engineer a robust, low-latency pipeline that ensures data integrity and high-performance querying:
* **Storage**: DuckDB-based columnar storage for millisecond-speed research over millions of ticks.
* **Integrity**: Strict Pydantic-based validation at the Silver layer.
* **Intelligence**: Feature-rich Gold layer including Market Structure (BOS/CHoCH) and Fairness Value Gaps (FVG).

## 🚀 Key Features
* **Unified Pipeline**: The same logic drives both **Live Trading** (`main.py`) and **Historical Backtesting** (`ingest_history.py`).
* **Scout & Sniper Strategy**: Advanced Smart Money Concepts (SMC) engine that detects structural breaks and executes precision entries into unfilled FVGs.
* **Hybrid Storage**: Bronze layer in partitioned Parquet files; Gold layer in a high-performance DuckDB feature store.

## 🛠️ How to Run the Project

### 1. Requirements
* **Python 3.11+**
* **DuckDB CLI** (optional, for manual SQL audits)

### 2. Setup environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Run Historical Ingestion & Strategy Audit
Download historical data and automatically run the strategy engine over it:
```bash
python ingest_history.py --start 2024-03-01 --end 2024-03-08
```

### 4. Run Live Trading Pipeline
Stream real-time data from MT5 directly into the DuckDB Gold layer:
```bash
python main.py
```

### 📈 Historical Analytics & Backtesting
The system provides a tick-by-tick event-loop engine for full simulation.
* **Check Strategy Decisions**:
  ```bash
  duckdb data/gold/goldstream.duckdb -s "SELECT * FROM trade_decisions LIMIT 10"
  ```

#### 🏗️ Medallion Architecture
- **Bronze**: Local raw partitioned Parquet (Source-specific).
- **Silver**: Normalized, validated Pydantic models.
- **Gold**: DuckDB Feature Store with RSI, ATR, BOS, CHoCH, and FVG detection.
- **Backtesting**: Tick-by-tick results with spread simulation and trailing stops.
