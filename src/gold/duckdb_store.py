"""
Gold Layer — DuckDBStore
Manages a local DuckDB database that acts as the "Backtest-Ready" feature store.
Holds unified ticks and their computed features (RSI, ATR, Liquidity levels).

DuckDB is chosen for:
  • Zero-config embedded SQL (no server needed)
  • Lightning-fast columnar scans on tens of millions of tick rows
  • First-class Parquet / Arrow interop
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd

from src.validation.validators import UnifiedTick

logger = logging.getLogger(__name__)

# Default path relative to project root
DEFAULT_DB_PATH = "data/gold/goldstream.duckdb"


class DuckDBStore:
    """
    Manages the DuckDB Gold layer database.

    Usage:
        store = DuckDBStore()
        store.init_schema()
        store.insert_unified_ticks(ticks)
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con: duckdb.DuckDBPyConnection = duckdb.connect(str(self.db_path))
        logger.info(f"[DuckDBStore] Connected to {self.db_path}")

    # ── Schema ────────────────────────────────────────────────────────────────

    def init_schema(self) -> None:
        """Create tables if they don't already exist."""
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS unified_ticks (
                timestamp_utc   TIMESTAMPTZ NOT NULL,
                symbol          VARCHAR     NOT NULL,
                bid             DOUBLE      NOT NULL,
                ask             DOUBLE      NOT NULL,
                volume          DOUBLE,
                source          VARCHAR,
                PRIMARY KEY (timestamp_utc, symbol, source)
            )
        """)

        self._con.execute("""
            CREATE TABLE IF NOT EXISTS tick_features (
                timestamp_utc   TIMESTAMPTZ NOT NULL,
                symbol          VARCHAR     NOT NULL,
                bid             DOUBLE,
                ask             DOUBLE,
                volume          DOUBLE,
                source          VARCHAR,
                -- OHLC candle fields (aligned to 5-min bar)
                bar_open    DOUBLE,
                bar_high    DOUBLE,
                bar_low     DOUBLE,
                bar_close   DOUBLE,
                -- Technical indicators
                rsi_14      DOUBLE,
                atr_14      DOUBLE,
                -- Liquidity
                liq_level   DOUBLE,
                liq_type    VARCHAR,   -- 'swing_high' | 'swing_low' | 'round_number'
                PRIMARY KEY (timestamp_utc, symbol, source)
            )
        """)

        self._con.execute("""
            CREATE INDEX IF NOT EXISTS idx_features_symbol_time
            ON tick_features (symbol, timestamp_utc)
        """)
        logger.info("[DuckDBStore] Schema initialised")

    # ── Writes ────────────────────────────────────────────────────────────────

    def insert_unified_ticks(self, ticks: Iterable[UnifiedTick], batch_size: int = 50_000) -> int:
        """
        Bulk-insert UnifiedTick objects using Arrow / DuckDB fast path.
        Uses INSERT OR IGNORE semantics to be idempotent.

        Returns the total number of rows inserted.
        """
        batch: list[dict] = []
        total = 0

        def _flush(rows: list[dict]) -> None:
            nonlocal total
            df = pd.DataFrame(rows)
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
            self._con.execute("""
                INSERT OR IGNORE INTO unified_ticks
                SELECT * FROM df
            """)
            total += len(rows)

        for tick in ticks:
            batch.append(tick.model_dump())
            if len(batch) >= batch_size:
                _flush(batch)
                batch.clear()
                logger.debug(f"[DuckDBStore] Flushed batch, total so far: {total}")

        if batch:
            _flush(batch)

        logger.info(f"[DuckDBStore] Inserted {total} rows into unified_ticks")
        return total

    def upsert_features(self, df: pd.DataFrame) -> None:
        """
        Upsert a features DataFrame (from FeatureEngineer) into tick_features.
        Any existing rows with the same (timestamp_utc, symbol, source) are replaced.
        """
        if df.empty:
            return
        # DELETE + INSERT pattern (DuckDB doesn't have native UPSERT for all cases)
        self._con.execute("DELETE FROM tick_features WHERE symbol = ?", [df["symbol"].iloc[0]])
        self._con.execute("INSERT INTO tick_features SELECT * FROM df")
        logger.info(f"[DuckDBStore] Upserted {len(df)} rows into tick_features")

    # ── Queries ───────────────────────────────────────────────────────────────

    def query_features(
        self,
        symbol: str,
        from_dt: datetime,
        to_dt: datetime,
    ) -> pd.DataFrame:
        """Return tick_features rows for `symbol` in [from_dt, to_dt]."""
        return self._con.execute("""
            SELECT *
            FROM tick_features
            WHERE symbol = ?
              AND timestamp_utc >= ?
              AND timestamp_utc <= ?
            ORDER BY timestamp_utc ASC
        """, [symbol, from_dt, to_dt]).df()

    def get_liquidity_levels(
        self,
        symbol: str,
        n_levels: int = 10,
    ) -> pd.DataFrame:
        """Return the N most significant liquidity levels for a symbol."""
        return self._con.execute("""
            SELECT DISTINCT liq_level, liq_type, COUNT(*) AS touches
            FROM tick_features
            WHERE symbol = ?
              AND liq_level IS NOT NULL
            GROUP BY liq_level, liq_type
            ORDER BY touches DESC
            LIMIT ?
        """, [symbol, n_levels]).df()

    def get_tick_count(self, table: str = "unified_ticks") -> int:
        """Quick row count for a given table."""
        return self._con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._con.close()
        logger.info("[DuckDBStore] Connection closed")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
