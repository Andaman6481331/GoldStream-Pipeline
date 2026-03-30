"""
Gold Layer Runner (DuckDB version)
Reads enriched tick_features rows (produced by FeatureEngineer v3),
builds a ScoutSniperContext per tick, calls make_decision, and persists
the result to trade_decisions in DuckDB.
"""

import logging
import os
import pandas as pd
from dotenv import load_dotenv

from src.bot.strategy_scout_sniper import make_decision, build_context_from_row, ScoutSniperContext
from src.backtest.backtest_engine import Action
from src.gold.duckdb_store import DuckDBStore

load_dotenv()
logger = logging.getLogger(__name__)

async def run_gold_layer(db_path: str = "data/gold/goldstream.duckdb") -> None:
    """
    Main runner for the Gold layer using DuckDB.
    1. Connects to DuckDB.
    2. Queries unprocessed ticks from tick_features.
    3. Runs Scout & Sniper strategy.
    4. Persists OPEN_LONG / OPEN_SHORT decisions.
    """
    logger.info(f"Gold layer runner (DuckDB) starting with {db_path}...")

    try:
        with DuckDBStore(db_path=db_path) as store:
            # ── Pull only ticks that are worth evaluating ─────────────────────
            # We look for ticks in tick_features that don't have a decision yet.
            query = """
                SELECT f.*
                FROM tick_features f
                LEFT JOIN trade_decisions t
                    ON f.timestamp_utc = t.tick_time
                    AND f.symbol = t.symbol
                WHERE t.tick_time IS NULL              -- not yet processed
                  AND f.atr_15_15m IS NOT NULL         -- health check
                  AND (f.bos_detected_15m = TRUE OR f.choch_detected_15m = TRUE) -- structure trigger
                ORDER BY f.timestamp_utc ASC
                LIMIT 1000
            """
            rows_df = store._con.execute(query).df()
            
            if rows_df.empty:
                logger.debug("No qualifying ticks (with structure breaks) to process")
                return

            logger.info(f"Processing {len(rows_df)} qualifying ticks...")

            saved = 0
            skipped = 0

            # Track last acted-on 15m signal to prevent multi-fire triggers on every tick
            # inside the same candle.
            _last_bos_15m_bar = None
            
            # Track if T1 just hit SL (recovery relay)
            _t1_stopped_at_loss = False
            
            # For audit logging, we treat active trade slots as local boolean flags
            # to ensure that build_context_from_row knows if a trade is "on".
            _t1_active = False
            _t2_active = False

            for _, row_series in rows_df.iterrows():
                row = row_series.to_dict()
                
                ts = pd.to_datetime(row["timestamp_utc"])
                current_15m_bar = ts.replace(minute=(ts.minute // 15) * 15, second=0, microsecond=0)
                
                # Reset structural signal deduplicator + recovery relay on new bar
                if _last_bos_15m_bar != current_15m_bar:
                    _last_bos_15m_bar = None
                    _t1_stopped_at_loss = False

                # Build context
                ctx: ScoutSniperContext = build_context_from_row(row, t1_stopped_at_loss=_t1_stopped_at_loss)
                ctx.t1_active = _t1_active
                ctx.t2_active = _t2_active

                # If we acted on a signal in this bar, suppress it from context
                if _last_bos_15m_bar == current_15m_bar:
                    ctx.bos_detected_15m = False
                    ctx.choch_detected_15m = False

                # Get decision
                result: DecisionSummary = make_decision(ctx)
                action: Action = result.action
                reason: str = result.reason

                # Update pseudo-state for logging
                if action in (Action.OPEN_T1_LONG, Action.OPEN_T1_SHORT):
                    _t1_active = True
                    _last_bos_15m_bar = current_15m_bar
                elif action == Action.CLOSE_T1:
                    _t1_active = False
                    # Audit assumption: if someone manually closes T1, it triggers Sniper possibility
                    _t1_stopped_at_loss = True 
                elif action in (Action.OPEN_T2_LONG, Action.OPEN_T2_SHORT):
                    _t2_active = True
                    _t1_stopped_at_loss = False
                elif action == Action.CLOSE_T2:
                    _t2_active = False
                elif action == Action.CLOSE_ALL:
                    _t1_active = False
                    _t2_active = False
                
                # Skip HOLDs
                if action == Action.HOLD:
                    skipped += 1
                    continue

                decision_str = action.value

                # Prepare decision record
                decision_record = {
                    "symbol": row["symbol"],
                    "tick_time": row["timestamp_utc"],
                    "decision": decision_str,
                    "reason": reason,
                    "score": result.metadata.get("score"),
                    "mid": ctx.mid,
                    "bid": ctx.bid,
                    "ask": ctx.ask,
                    "session": ctx.session,
                    "rsi_14": ctx.rsi_14,
                    "price_position": ctx.price_position,
                    "fvg_high": ctx.fvg_high,
                    "fvg_low": ctx.fvg_low,
                    "fvg_side": ctx.fvg_side,
                    "fvg_filled": ctx.fvg_filled,
                    "fvg_age_bars": ctx.fvg_age_bars,
                    "atr_20_1m": ctx.atr_20_1m,
                    "atr_15_15m": ctx.atr_15_15m,
                    "prev_day_high": ctx.prev_day_high,
                    "prev_day_low": ctx.prev_day_low,
                    "current_session_high": ctx.current_session_high,
                    "current_session_low": ctx.current_session_low,
                    "prev_session_high": ctx.prev_session_high,
                    "prev_session_low": ctx.prev_session_low,
                    "session_boundary": ctx.session_boundary,
                    "n_confirmed_swing_highs_15m": ctx.n_confirmed_swing_highs_15m,
                    "n_confirmed_swing_lows_15m": ctx.n_confirmed_swing_lows_15m,
                    "fvg_timestamp": ctx.fvg_timestamp,
                    "smc_trend_15m": ctx.smc_trend_15m,
                    "hh_15m": ctx.hh_15m,
                    "ll_15m": ctx.ll_15m,
                    "strong_low_15m": ctx.strong_low_15m,
                    "strong_high_15m": ctx.strong_high_15m,
                    "bos_detected_15m": ctx.bos_detected_15m,
                    "choch_detected_15m": ctx.choch_detected_15m,
                    "market_bias_4h": ctx.market_bias_4h,
                    "liq_swept": ctx.liq_swept,
                    "liq_side": ctx.liq_side
                }

                # Persist to DuckDB
                store.insert_trade_decision(decision_record)
                saved += 1

                logger.info(
                    f"🎯 {decision_str:10s} | "
                    f"BID={ctx.bid:.5f} | "
                    f"Trend={ctx.smc_trend_15m} | "
                    f"FVG Side={ctx.fvg_side} | "
                    f"Session={ctx.session}"
                )

            logger.info(
                f"Gold layer complete — "
                f"{saved} decisions saved, {skipped} HOLDs skipped"
            )

    except Exception as exc:
        logger.error(f"Gold layer runner error: {exc}", exc_info=True)
        raise

if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_gold_layer())