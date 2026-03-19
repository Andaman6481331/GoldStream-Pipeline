### GoldStream-Pipeline
Near Real-Time data engineering pipeline designed to ingest, validate, and store high-frequency XAUUSD (Gold) market data from MetaTrader 5. Unlike a standard trading script, this system focuses on data reliability, integrity, and scalability, following industry-standard patterns such as the Medallion Architecture (Bronze, Silver, and Gold layers).

## 🎯 Project Objective
The primary goal is to engineer a robust infrastructure that converts raw, unpredictable financial data into a "production-ready" dataset for automated trading and analysis. Key objectives include:
* Real-time Ingestion: Implementing an asynchronous Python-based collector for low-latency tick data.
* Data Integrity: Automating data quality checks and validation using Pydantic to prevent "dirty" data from entering the database.
* Scalable Storage: Architecting a containerized PostgreSQL database (optimized for time-series data) to ensure long-term data persistence and accessibility.

## 🔭 Project Scope
This project focuses on the infrastructure and flow of data rather than just the final trading outcome. 

# In-Scope:
* Development of a multi-stage ETL/ELT pipeline.
* Containerization of the entire stack using Docker for consistent deployment.
* Implementation of automated logging and monitoring for system health.

# Out-of-Scope:
* Developing complex machine learning models (this is handled in the "Gold" or downstream analytics layer).
* Direct integration with live brokerage accounts for real-money execution (currently focused on MT5 backtest/demo feeds).

## 🚀 How to Run the Project

### 1. Requirements

Ensure you have the following installed on your system:
- **Python 3.8+**
- **Docker** and **Docker Compose**

### 2. Create and Activate a Virtual Environment

First, create a virtual environment to manage your Python dependencies.

**On Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

With your virtual environment active, install the required Python packages:

```bash
pip install -r requirements.txt
```

### 4. Start Infrastructure (Docker)

Start the required infrastructure (PostgreSQL database and Adminer for database management) using Docker Compose:

```bash
docker-compose up -d postgres adminer
```
*Note: Make sure Docker Desktop is running before executing this command.*

### 5. Run the Application

Once the database is up and running, you can start the main Python pipeline:

```bash
python main.py
```