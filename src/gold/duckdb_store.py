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
from typing import Iterable, Optional

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

    def __init__(self, db_path: str = DEFAULT_DB_PATH, read_only: bool = False):
        self.db_path = Path(db_path)
        if not read_only:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con: duckdb.DuckDBPyConnection = duckdb.connect(str(self.db_path), read_only=read_only)
        logger.info(f"[DuckDBStore] Connected to {self.db_path} (read_only={read_only})")

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
                volume_usd      DOUBLE,
                source          VARCHAR,
                PRIMARY KEY (timestamp_utc, symbol, source)
            )
        """)

        # ── Tick Features (Gold) ──────────────────────────────────────────────
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS tick_features (
                timestamp_utc   TIMESTAMPTZ NOT NULL,
                symbol          VARCHAR     NOT NULL,
                bid             DOUBLE,
                ask             DOUBLE,
                mid             DOUBLE,
                volume          DOUBLE,
                volume_usd      DOUBLE,
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
                liq_side    VARCHAR,
                liq_tf      VARCHAR,
                liq_score   DOUBLE,
                liq_confirmed BOOLEAN,
                liq_swept     BOOLEAN,
                dist_to_nearest_high DOUBLE,
                dist_to_nearest_low  DOUBLE,
                session     VARCHAR,

                -- Market Structure & FVG Columns
                structure_direction  VARCHAR,
                bos_detected         BOOLEAN,
                choch_detected       BOOLEAN,
                fvg_high             DOUBLE,
                fvg_low              DOUBLE,
                fvg_side             VARCHAR,
                fvg_timestamp        TIMESTAMPTZ,
                fvg_filled           BOOLEAN,
                fvg_age_bars         INTEGER,
                price_position       VARCHAR,

                -- NEW-P1: SMC ATR indicators
                atr_20_1m            DOUBLE,
                atr_15_15m           DOUBLE,
                -- NEW-P1: Session levels
                prev_day_high        DOUBLE,
                prev_day_low         DOUBLE,
                current_session_high DOUBLE,
                current_session_low  DOUBLE,
                prev_session_high    DOUBLE,
                prev_session_low     DOUBLE,
                session_boundary     BOOLEAN,
                -- NEW-P1: Swing history counts
                n_confirmed_swing_highs_15m INTEGER,
                n_confirmed_swing_lows_15m  INTEGER,
                -- Phase 2 SMC: Structural Nodes (15m)
                smc_trend_15m        VARCHAR,
                hh_15m               DOUBLE,
                ll_15m               DOUBLE,
                strong_low_15m       DOUBLE,
                strong_high_15m      DOUBLE,
                bos_detected_15m     BOOLEAN,
                choch_detected_15m   BOOLEAN,
                bos_up_15m           BOOLEAN,
                bos_down_15m         BOOLEAN,
                choch_up_15m         BOOLEAN,
                choch_down_15m       BOOLEAN,
                is_swing_high_15m    BOOLEAN,
                is_swing_low_15m     BOOLEAN,
                market_bias_4h       VARCHAR,

                PRIMARY KEY (timestamp_utc, symbol, source)
            )
        """)

        self._con.execute("""
            CREATE TABLE IF NOT EXISTS trade_decisions (
                symbol          VARCHAR     NOT NULL,
                tick_time       TIMESTAMPTZ NOT NULL,
                decision        VARCHAR     NOT NULL,
                reason          VARCHAR,
                score           INTEGER,
                -- Context fields
                mid             DOUBLE,
                bid             DOUBLE,
                ask             DOUBLE,
                session         VARCHAR,
                fvg_high        DOUBLE,
                fvg_low         DOUBLE,
                fvg_side        VARCHAR,
                fvg_filled      BOOLEAN,
                fvg_age_bars    INTEGER,
                -- Phase 1 SMC Extended Context
                atr_20_1m       DOUBLE,
                atr_15_15m      DOUBLE,
                prev_day_high   DOUBLE,
                prev_day_low    DOUBLE,
                current_session_high DOUBLE,
                current_session_low  DOUBLE,
                prev_session_high    DOUBLE,
                prev_session_low     DOUBLE,
                session_boundary     BOOLEAN,
                n_confirmed_swing_highs_15m INTEGER,
                n_confirmed_swing_lows_15m  INTEGER,
                fvg_timestamp        TIMESTAMPTZ,
                -- Phase 2 SMC: Structural Nodes (15m)
                smc_trend_15m        VARCHAR,
                hh_15m               DOUBLE,
                ll_15m               DOUBLE,
                strong_low_15m       DOUBLE,
                strong_high_15m      DOUBLE,
                bos_detected_15m     BOOLEAN,
                choch_detected_15m   BOOLEAN,
                bos_up_15m           BOOLEAN,
                bos_down_15m         BOOLEAN,
                choch_up_15m         BOOLEAN,
                choch_down_15m       BOOLEAN,
                is_swing_high_15m    BOOLEAN,
                is_swing_low_15m     BOOLEAN,
                market_bias_4h       VARCHAR,
                liq_swept            BOOLEAN,
                liq_side             VARCHAR,

                PRIMARY KEY (tick_time, symbol)
            )
        """)

        # ── Multi-Timeframe Candle Tables ─────────────────────────────────────
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS candles_1m (
                bar_time    TIMESTAMPTZ NOT NULL,
                symbol      VARCHAR     NOT NULL,
                source      VARCHAR     NOT NULL,
                bar_open    DOUBLE      NOT NULL,
                bar_high    DOUBLE      NOT NULL,
                bar_low     DOUBLE      NOT NULL,
                bar_close   DOUBLE      NOT NULL,
                atr_20_1m   DOUBLE,
                PRIMARY KEY (bar_time, symbol, source)
            )
        """)
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS candles_15m (
                bar_time            TIMESTAMPTZ NOT NULL,
                symbol              VARCHAR     NOT NULL,
                source              VARCHAR     NOT NULL,
                bar_open            DOUBLE      NOT NULL,
                bar_high            DOUBLE      NOT NULL,
                bar_low             DOUBLE      NOT NULL,
                bar_close           DOUBLE      NOT NULL,
                atr_15_15m          DOUBLE,
                smc_trend_15m       VARCHAR,
                hh_15m              DOUBLE,
                ll_15m              DOUBLE,
                strong_low_15m      DOUBLE,
                strong_high_15m     DOUBLE,
                bos_detected_15m    BOOLEAN,
                choch_detected_15m  BOOLEAN,
                -- Directional signals
                bos_up_15m          BOOLEAN,
                bos_down_15m        BOOLEAN,
                choch_up_15m        BOOLEAN,
                choch_down_15m      BOOLEAN,
                is_swing_high_15m   BOOLEAN,
                is_swing_low_15m    BOOLEAN,
                PRIMARY KEY (bar_time, symbol, source)
            )
        """)
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS candles_4h (
                bar_time        TIMESTAMPTZ NOT NULL,
                symbol          VARCHAR     NOT NULL,
                source          VARCHAR     NOT NULL,
                bar_open        DOUBLE      NOT NULL,
                bar_high        DOUBLE      NOT NULL,
                bar_low         DOUBLE      NOT NULL,
                bar_close       DOUBLE      NOT NULL,
                market_bias_4h  VARCHAR,
                PRIMARY KEY (bar_time, symbol, source)
            )
        """)

        self._con.execute("""
            CREATE INDEX IF NOT EXISTS idx_features_symbol_time
            ON tick_features (symbol, timestamp_utc)
        """)
        self._migrate_schema()
        logger.info("[DuckDBStore] Schema initialised")

    def _migrate_schema(self) -> None:
        """Add missing columns to existing tables if needed."""
        # unified_ticks migrations
        self._add_column_if_not_exists("unified_ticks", "volume_usd", "DOUBLE")
        
        # tick_features migrations
        self._add_column_if_not_exists("tick_features", "mid", "DOUBLE")
        self._add_column_if_not_exists("tick_features", "volume_usd", "DOUBLE")
        self._add_column_if_not_exists("tick_features", "liq_side", "VARCHAR")
        self._add_column_if_not_exists("tick_features", "liq_tf", "VARCHAR")
        self._add_column_if_not_exists("tick_features", "liq_score", "DOUBLE")
        self._add_column_if_not_exists("tick_features", "liq_confirmed", "BOOLEAN")
        self._add_column_if_not_exists("tick_features", "liq_swept", "BOOLEAN")
        self._add_column_if_not_exists("tick_features", "dist_to_nearest_high", "DOUBLE")
        self._add_column_if_not_exists("tick_features", "dist_to_nearest_low", "DOUBLE")
        self._add_column_if_not_exists("tick_features", "session", "VARCHAR")
        self._add_column_if_not_exists("tick_features", "price_position", "VARCHAR")

        # Phase 1 & 2: SMC Features & Structural Nodes
        self._add_column_if_not_exists("tick_features", "fvg_timestamp", "TIMESTAMPTZ")
        self._add_column_if_not_exists("tick_features", "atr_20_1m", "DOUBLE")
        self._add_column_if_not_exists("tick_features", "atr_15_15m", "DOUBLE")
        self._add_column_if_not_exists("tick_features", "prev_day_high", "DOUBLE")
        self._add_column_if_not_exists("tick_features", "prev_day_low", "DOUBLE")
        self._add_column_if_not_exists("tick_features", "current_session_high", "DOUBLE")
        self._add_column_if_not_exists("tick_features", "current_session_low", "DOUBLE")
        self._add_column_if_not_exists("tick_features", "prev_session_high", "DOUBLE")
        self._add_column_if_not_exists("tick_features", "prev_session_low", "DOUBLE")
        self._add_column_if_not_exists("tick_features", "session_boundary", "BOOLEAN")
        self._add_column_if_not_exists("tick_features", "n_confirmed_swing_highs_15m", "INTEGER")
        self._add_column_if_not_exists("tick_features", "n_confirmed_swing_lows_15m", "INTEGER")

        # Phase 2: 15m Structural Nodes
        self._add_column_if_not_exists("tick_features", "smc_trend_15m", "VARCHAR")
        self._add_column_if_not_exists("tick_features", "hh_15m", "DOUBLE")
        self._add_column_if_not_exists("tick_features", "ll_15m", "DOUBLE")
        self._add_column_if_not_exists("tick_features", "strong_low_15m", "DOUBLE")
        self._add_column_if_not_exists("tick_features", "strong_high_15m", "DOUBLE")
        self._add_column_if_not_exists("tick_features", "bos_detected_15m", "BOOLEAN")
        self._add_column_if_not_exists("tick_features", "choch_detected_15m", "BOOLEAN")
        self._add_column_if_not_exists("tick_features", "bos_up_15m", "BOOLEAN")
        self._add_column_if_not_exists("tick_features", "bos_down_15m", "BOOLEAN")
        self._add_column_if_not_exists("tick_features", "choch_up_15m", "BOOLEAN")
        self._add_column_if_not_exists("tick_features", "choch_down_15m", "BOOLEAN")
        self._add_column_if_not_exists("tick_features", "is_swing_high_15m", "BOOLEAN")
        self._add_column_if_not_exists("tick_features", "is_swing_low_15m", "BOOLEAN")
        self._add_column_if_not_exists("tick_features", "market_bias_4h", "VARCHAR")

        # candles_15m migrations
        self._add_column_if_not_exists("candles_15m", "bos_up_15m", "BOOLEAN")
        self._add_column_if_not_exists("candles_15m", "bos_down_15m", "BOOLEAN")
        self._add_column_if_not_exists("candles_15m", "choch_up_15m", "BOOLEAN")
        self._add_column_if_not_exists("candles_15m", "choch_down_15m", "BOOLEAN")
        self._add_column_if_not_exists("candles_15m", "is_swing_high_15m", "BOOLEAN")
        self._add_column_if_not_exists("candles_15m", "is_swing_low_15m", "BOOLEAN")

        self._add_column_if_not_exists("tick_features", "structure_direction", "VARCHAR")
        self._add_column_if_not_exists("tick_features", "bos_detected", "BOOLEAN")
        self._add_column_if_not_exists("tick_features", "choch_detected", "BOOLEAN")
        self._add_column_if_not_exists("tick_features", "fvg_high", "DOUBLE")
        self._add_column_if_not_exists("tick_features", "fvg_low", "DOUBLE")
        self._add_column_if_not_exists("tick_features", "fvg_side", "VARCHAR")
        self._add_column_if_not_exists("tick_features", "fvg_filled", "BOOLEAN")
        self._add_column_if_not_exists("tick_features", "fvg_age_bars", "INTEGER")

        # trade_decisions migrations
        self._add_column_if_not_exists("trade_decisions", "fvg_filled", "BOOLEAN")
        self._add_column_if_not_exists("trade_decisions", "fvg_age_bars", "INTEGER")
        self._add_column_if_not_exists("trade_decisions", "fvg_timestamp", "TIMESTAMPTZ")

    def _add_column_if_not_exists(self, table: str, column: str, dtype: str) -> None:
        """Helper to safely add a column to an existing table."""
        try:
            self._con.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {dtype}")
        except Exception as exc:
            logger.error(f"[DuckDBStore] Failed to migrate {table}.{column}: {exc}")

    # ── Writes ────────────────────────────────────────────────────────────────

    def insert_unified_ticks(self, ticks: Iterable[UnifiedTick], batch_size: int = 50_000) -> int:
        """
        Bulk-insert UnifiedTick objects using Arrow / DuckDB fast path.
        Uses INSERT OR IGNORE semantics to be idempotent.

        Returns the total number of rows inserted.
        """
        batch: list[dict] = []
        total_inserted = [0]

        def _flush(rows: list[dict]) -> None:
            df = pd.DataFrame(rows)
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
            self._con.execute("""
                INSERT OR IGNORE INTO unified_ticks 
                    (timestamp_utc, symbol, bid, ask, volume, volume_usd, source)
                SELECT 
                    timestamp_utc, symbol, bid, ask, volume, volume_usd, source
                FROM df
            """)
            total_inserted[0] += len(rows)

        for tick in ticks:
            batch.append(tick.model_dump())
            if len(batch) >= batch_size:
                _flush(batch)
                batch.clear()
                logger.debug(f"[DuckDBStore] Flushed batch, total so far: {total_inserted[0]}")

        if batch:
            _flush(batch)

        logger.info(f"[DuckDBStore] Inserted {total_inserted[0]} rows into unified_ticks")
        return total_inserted[0]

    def upsert_features(self, df: pd.DataFrame) -> None:
        """
        Upsert a features DataFrame (from FeatureEngineer) into tick_features.
        Any existing rows with the same (timestamp_utc, symbol, source) are replaced.
        """
        if df.empty:
            return
        # DELETE + INSERT pattern (DuckDB doesn't have native UPSERT for all cases)
        cols = [
            "timestamp_utc", "symbol", "bid", "ask", "mid", "volume", "volume_usd", "source",
            "bar_open", "bar_high", "bar_low", "bar_close",
            "rsi_14", "atr_14",
            "liq_level", "liq_type", "liq_side", "liq_tf",
            "liq_score", "liq_confirmed", "liq_swept",
            "dist_to_nearest_high", "dist_to_nearest_low",
            "session",

            "structure_direction", "bos_detected", "choch_detected",
            "fvg_high", "fvg_low", "fvg_side", "fvg_timestamp", "fvg_filled",
            "fvg_age_bars", "price_position",
            "atr_20_1m", "atr_15_15m",
            "prev_day_high", "prev_day_low", "current_session_high", "current_session_low",
            "prev_session_high", "prev_session_low", "session_boundary",
            "n_confirmed_swing_highs_15m", "n_confirmed_swing_lows_15m",
            "smc_trend_15m", "hh_15m", "ll_15m", "strong_low_15m", "strong_high_15m",
            "bos_detected_15m", "choch_detected_15m", "market_bias_4h",
            "bos_up_15m", "bos_down_15m", "choch_up_15m", "choch_down_15m",
            "is_swing_high_15m", "is_swing_low_15m"
        ]
        # Only use columns that exist in the DataFrame
        available = [c for c in cols if c in df.columns]
        col_names = ", ".join(available)
        placeholders = ", ".join([f"{c}" for c in available])
        
        self._con.execute(f"DELETE FROM tick_features WHERE symbol = ?", [df["symbol"].iloc[0]])
        self._con.execute(f"INSERT INTO tick_features ({col_names}) SELECT {col_names} FROM df")
        logger.info(f"[DuckDBStore] Upserted {len(df)} rows into tick_features")

    def upsert_candles(self, table: str, df: pd.DataFrame) -> None:
        """
        Upsert a candle DataFrame into one of the candle tables (candles_1m, 15m, 4h).
        Any existing rows for the same symbol are replaced.
        Uses INSERT INTO ... BY NAME to correctly map DataFrame columns to table schema.
        """
        if df.empty:
            return
        
        # Ensure bar_time is a timestamp
        if "bar_time" in df.columns:
            df["bar_time"] = pd.to_datetime(df["bar_time"], utc=True)

        symbol = df["symbol"].iloc[0]
        self._con.execute(f"DELETE FROM {table} WHERE symbol = ?", [symbol])
        
        # Only select columns that exist in the target table to avoid extra columns in DF error
        target_cols_q = self._con.execute(f"DESCRIBE {table}").fetchall()
        target_cols = [c[0] for c in target_cols_q]
        df_cols = [c for c in df.columns if c in target_cols]
        
        # Perform insertion by name
        self._con.execute(f"INSERT INTO {table} ({', '.join(df_cols)}) SELECT {', '.join(df_cols)} FROM df")
        logger.info(f"[DuckDBStore] Upserted {len(df)} rows into {table}")

    def insert_trade_decision(self, decision_data: dict) -> None:
        """Persist a single trade decision into the trade_decisions table."""
        cols = [
            "symbol", "tick_time", "decision", "reason", "score",
            "mid", "bid", "ask", "session",
            "fvg_high", "fvg_low", "fvg_side", "fvg_filled", "fvg_age_bars", "fvg_timestamp",
            "atr_20_1m", "atr_15_15m",
            "prev_day_high", "prev_day_low",
            "current_session_high", "current_session_low",
            "prev_session_high", "prev_session_low", "session_boundary",
            "n_confirmed_swing_highs_15m", "n_confirmed_swing_lows_15m",
            "smc_trend_15m", "hh_15m", "ll_15m", "strong_low_15m", "strong_high_15m",
            "bos_detected_15m", "choch_detected_15m", "market_bias_4h",
            "bos_up_15m", "bos_down_15m", "choch_up_15m", "choch_down_15m",
            "is_swing_high_15m", "is_swing_low_15m",
            "liq_swept", "liq_side"
        ]
        # Ensure tick_time is a datetime object for DuckDB
        if isinstance(decision_data["tick_time"], str):
             decision_data["tick_time"] = pd.to_datetime(decision_data["tick_time"])

        col_names = ", ".join(cols)
        placeholders = ", ".join(["?" for _ in cols])
        values = [decision_data.get(c) for c in cols]

        self._con.execute(f"""
            INSERT OR IGNORE INTO trade_decisions ({col_names})
            VALUES ({placeholders})
        """, values)

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

    def query_candles_at(
        self,
        table: str,
        symbol: str,
        ts: datetime,
    ) -> Optional[dict]:
        """
        Return the single 1m candle whose bar_time <= ts.
        Used by BacktestEngine to get entry_candle_high/low for Point 1 calculation.
        
        Args:
            table  : "candles_1m" | "candles_15m" | "candles_4h"
            symbol : e.g. "XAUUSD"
            ts     : tick timestamp at trade entry moment
        """
        result = self._con.execute(f"""
            SELECT bar_open, bar_high, bar_low, bar_close
            FROM {table}
            WHERE symbol = ?
            AND bar_time <= ?
            ORDER BY bar_time DESC
            LIMIT 1
        """, [symbol, ts]).fetchone()

        if result is None:
            return None

        return {
            "bar_open":  result[0],
            "bar_high":  result[1],
            "bar_low":   result[2],
            "bar_close": result[3],
        }

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

    def query_decisions(self, symbol: str) -> pd.DataFrame:
        """Return all trade decisions for a symbol."""
        return self._con.execute("""
            SELECT * FROM trade_decisions 
            WHERE symbol = ? 
            ORDER BY tick_time ASC
        """, [symbol]).df()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._con.close()
        logger.info("[DuckDBStore] Connection closed")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
