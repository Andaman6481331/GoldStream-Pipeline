"""
Gold Layer Runner (DuckDB version)
Reads enriched tick_features rows (produced by FeatureEngineer v3),
builds a ScoutSniperContext per tick, calls make_decision, and persists
the result to trade_decisions in DuckDB.
"""

import logging
import os
import pandas as pd
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

from src.bot.strategy_scout_sniper import (
    make_decision,
    build_context_from_row,
    ScoutSniperContext,
    DecisionSummary,     # was missing — caused NameError on type annotation
)
from src.backtest.backtest_engine import Action
from src.gold.duckdb_store import DuckDBStore

load_dotenv()
logger = logging.getLogger(__name__)


async def run_gold_layer(
    db_path: str = "data/gold/goldstream.duckdb",
    from_dt: Optional[datetime] = None,
    to_dt:   Optional[datetime] = None,
    limit:   Optional[int]      = 1000
) -> None:
    """
    Main runner for the Gold layer using DuckDB.
    1. Connects to DuckDB.
    2. Queries unprocessed ticks from tick_features.
       - If from_dt/to_dt is provided, restricts to that range.
       - If from_dt is None, starts from the last audited tick (resume).
    3. Runs Scout & Sniper strategy.
    4. Persists all decisions to trade_decisions.
    """
    logger.info(f"Gold layer runner (DuckDB) starting with {db_path}...")

    try:
        with DuckDBStore(db_path=db_path) as store:
            # ── 1. Determine Range & Persistence ──────────────────────────────
            applied_from = from_dt
            if applied_from is None:
                # Resume logic: find the latest audited tick in the DB
                last_tick = store._con.execute("SELECT MAX(tick_time) FROM trade_decisions").fetchone()[0]
                if last_tick:
                    applied_from = last_tick
                    logger.info(f"[Audit] Resuming from last audited tick: {applied_from}")

            # ── 2. Build Query ────────────────────────────────────────────────
            query = """
                SELECT f.*
                FROM tick_features f
                LEFT JOIN trade_decisions t
                    ON f.timestamp_utc = t.tick_time
                    AND f.symbol = t.symbol
                WHERE t.tick_time IS NULL
                  AND f.atr_15_15m IS NOT NULL
                  AND (f.bos_detected_15m = TRUE OR f.choch_detected_15m = TRUE)
            """
            params = []
            if applied_from:
                query += " AND f.timestamp_utc >= ?"
                params.append(applied_from)
            if to_dt:
                query += " AND f.timestamp_utc <= ?"
                params.append(to_dt)
            
            query += " ORDER BY f.timestamp_utc ASC"
            
            if limit and limit > 0:
                query += f" LIMIT {limit}"

            rows_df = store._con.execute(query, params).df()

            if rows_df.empty:
                logger.debug("No qualifying ticks to process")
                return

            logger.info(f"Processing {len(rows_df)} qualifying ticks...")

            saved = 0

            # ── Engine-relay state ────────────────────────────────────────
            # These mirror the engine's internal state and must only be
            # mutated at the correct lifecycle events (not on bar boundaries).

            _t1_active          = False
            _t2_active          = False

            # True ONLY from T1 real loss until T2 resolves.
            # Reset on: T2 open, new T1 open, T2 timeout/cancel.
            # NOT reset on bar boundaries.
            _t1_stopped_at_loss = False

            # BOS context stored at T1 fire time — relayed for T2 Gate 3.
            _bos_direction:    str | None = None
            _bos_time_ms:      int | None = None
            _r_dynamic_at_bos: int | None = None

            # BOS deduplication — only suppresses after a trade was actually opened.
            # Never reset on a bare bar boundary.
            _bos_taken_bar = None

            for _, row_series in rows_df.iterrows():
                row = row_series.to_dict()

                ts = pd.to_datetime(row["timestamp_utc"])
                current_15m_bar = ts.replace(
                    minute=(ts.minute // 15) * 15, second=0, microsecond=0
                )

                # Build context — relay engine state as kwargs
                ctx: ScoutSniperContext = build_context_from_row(
                    row,
                    t1_stopped_at_loss=_t1_stopped_at_loss,
                    t1_active=_t1_active,
                    t2_active=_t2_active,
                    bos_direction=_bos_direction,
                    bos_time_ms=_bos_time_ms,
                    r_dynamic_at_bos=_r_dynamic_at_bos,
                )

                # BOS dedup — suppress only if a trade was already taken in this bar
                if _bos_taken_bar == current_15m_bar:
                    ctx.bos_detected_15m  = False
                    ctx.choch_detected_15m = False
                    ctx.bos_up_15m         = False
                    ctx.bos_down_15m       = False
                    ctx.choch_up_15m       = False
                    ctx.choch_down_15m     = False

                result: DecisionSummary = make_decision(ctx)
                action: Action          = result.action
                reason: str             = result.reason

                # ── Update relay state based on decision ──────────────────

                if action == Action.OPEN_T1_LONG:
                    _t1_active          = True
                    _t1_stopped_at_loss = False
                    _bos_taken_bar      = current_15m_bar
                    # Store BOS context for T2 Gate 3 relay
                    _bos_direction    = "bull"
                    _bos_time_ms      = int(ts.timestamp() * 1000)
                    _r_dynamic_at_bos = _safe_int(row.get("r_dynamic"))

                elif action == Action.OPEN_T1_SHORT:
                    _t1_active          = True
                    _t1_stopped_at_loss = False
                    _bos_taken_bar      = current_15m_bar
                    _bos_direction    = "bear"
                    _bos_time_ms      = int(ts.timestamp() * 1000)
                    _r_dynamic_at_bos = _safe_int(row.get("r_dynamic"))

                elif action == Action.CLOSE_T1:
                    # In the gold layer runner we don't know T1's PnL or whether
                    # Point 1 was hit — that's the backtest engine's job.
                    # Conservative approach: don't arm T2 from a strategy-initiated
                    # close since we can't verify it was a real loss.
                    _t1_active = False

                elif action in (Action.OPEN_T2_LONG, Action.OPEN_T2_SHORT):
                    _t2_active          = True
                    _t1_stopped_at_loss = False   # recovery sequence consumed

                elif action == Action.CLOSE_T2:
                    _t2_active = False

                elif action == Action.CLOSE_ALL:
                    _t1_active = False
                    _t2_active = False

                # Persist decision record
                decision_record = {
                    "symbol":                    row["symbol"],
                    "tick_time":                 row["timestamp_utc"],
                    "decision":                  action.value,
                    "reason":                    reason,
                    "score":                     result.metadata.get("total_score"),
                    "mid":                       ctx.mid,
                    "bid":                       ctx.bid,
                    "ask":                       ctx.ask,
                    "session":                   ctx.session,
                    "fvg_high":                  ctx.fvg_high,
                    "fvg_low":                   ctx.fvg_low,
                    "fvg_side":                  ctx.fvg_side,
                    "fvg_filled":                ctx.fvg_filled,
                    "fvg_age_bars":              ctx.fvg_age_bars,
                    "atr_20_1m":                 ctx.atr_20_1m,
                    "atr_15_15m":                ctx.atr_15_15m,
                    "prev_day_high":             ctx.prev_day_high,
                    "prev_day_low":              ctx.prev_day_low,
                    "current_session_high":      ctx.current_session_high,
                    "current_session_low":       ctx.current_session_low,
                    "prev_session_high":         ctx.prev_session_high,
                    "prev_session_low":          ctx.prev_session_low,
                    "session_boundary":          ctx.session_boundary,
                    "n_confirmed_swing_highs_15m": ctx.n_confirmed_swing_highs_15m,
                    "n_confirmed_swing_lows_15m":  ctx.n_confirmed_swing_lows_15m,
                    "fvg_timestamp":             ctx.fvg_timestamp,
                    "smc_trend_15m":             ctx.smc_trend_15m,
                    "hh_15m":                    ctx.hh_15m,
                    "ll_15m":                    ctx.ll_15m,
                    "strong_low_15m":            ctx.strong_low_15m,
                    "strong_high_15m":           ctx.strong_high_15m,
                    "bos_detected_15m":          ctx.bos_detected_15m,
                    "choch_detected_15m":        ctx.choch_detected_15m,
                    "bos_up_15m":                ctx.bos_up_15m,
                    "bos_down_15m":              ctx.bos_down_15m,
                    "choch_up_15m":              ctx.choch_up_15m,
                    "choch_down_15m":            ctx.choch_down_15m,
                    "is_swing_high_15m":         ctx.is_swing_high_15m,
                    "is_swing_low_15m":          ctx.is_swing_low_15m,
                    "market_bias_4h":            ctx.market_bias_4h,
                    "liq_swept":                 ctx.liq_swept,
                    "liq_side":                  ctx.liq_side,
                    "liq_tier":                  ctx.liq_tier,
                    # BOS relay context — for audit
                    "bos_direction":             _bos_direction,
                    "bos_time_ms":               _bos_time_ms,
                }

                store.insert_trade_decision(decision_record)
                saved += 1

                logger.debug(
                    f"{'🎯' if action != Action.HOLD else '·'} "
                    f"{action.value:15s} | "
                    f"Reason={reason:25s} | "
                    f"BID={ctx.bid:.5f} | "
                    f"Bias={ctx.market_bias_4h} | "
                    f"BOS_up={ctx.bos_up_15m} BOS_dn={ctx.bos_down_15m} | "
                    f"Session={ctx.session}"
                )

            logger.info(f"Gold layer complete — {saved} decisions saved")

    except Exception as exc:
        logger.error(f"Gold layer runner error: {exc}", exc_info=True)
        raise


def _safe_int(val) -> int | None:
    try:
        if val is None:
            return None
        return int(float(val))
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_gold_layer())