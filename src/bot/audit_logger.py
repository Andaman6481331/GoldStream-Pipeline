"""
Gold Layer Runner
Reads enriched tick_features rows (produced by FeatureEngineer v2),
builds a full SMCContext per tick, calls make_decision, and persists
the result to trade_decisions.

Key fixes vs previous version:
  - Reads from tick_features (not cleaned_ticks)
  - Pre-filters at SQL level: confirmed, unswept, scored, session-gated
  - Passes full SMCContext to make_decision (not just rsi/ema/bid)
  - INSERT writes all relevant v2 columns to trade_decisions
  - Logs decision reason fields for post-analysis
"""

import asyncpg
import asyncio
import os
import logging
from dotenv import load_dotenv

from src.bot.strategy import make_decision, build_context_from_row, SMCContext
from src.backtesting.backtest_engine import Action

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ── DB connection ─────────────────────────────────────────────────────────────

async def get_db_connection() -> asyncpg.Connection:
    return await asyncpg.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT")),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


# ── Main runner ───────────────────────────────────────────────────────────────

async def run_gold_layer() -> None:
    conn = await get_db_connection()
    logger.info("Gold layer runner starting...")

    try:
        # ── Pull only ticks that are worth evaluating ─────────────────────
        # Pre-filter at DB level to avoid loading noise into Python.
        # liq_confirmed / liq_swept / liq_score filter here means
        # make_decision's gate checks are a safety net, not the first line.
        rows = await conn.fetch("""
            SELECT f.*
            FROM tick_features f
            LEFT JOIN trade_decisions t
                ON f.timestamp_utc = t.tick_time
                AND f.symbol = t.symbol
            WHERE t.tick_time IS NULL              -- not yet processed
              AND f.liq_confirmed = TRUE           -- structure break proven
              AND f.liq_swept     = FALSE          -- level not yet consumed
              AND f.liq_score     >= 4             -- meaningful confluence
              AND f.session IN ('killzone', 'london')
              AND f.rsi_14  IS NOT NULL
              AND f.atr_14  IS NOT NULL
            ORDER BY f.timestamp_utc ASC
            LIMIT 500
        """)

        if not rows:
            logger.info("No qualifying ticks to process")
            return

        logger.info(f"Processing {len(rows)} qualifying ticks...")

        saved = 0
        skipped = 0

        for row in rows:
            # Build full SMC context from the row
            ctx: SMCContext = build_context_from_row(dict(row))

            # Get decision
            action: Action = make_decision(ctx)
            decision_str = action.value   # "OPEN_LONG" | "OPEN_SHORT" | "HOLD"

            # Skip HOLDs — no point storing non-events
            if action == Action.HOLD:
                skipped += 1
                continue

            # Persist the decision with full context for post-analysis
            await conn.execute("""
                INSERT INTO trade_decisions (
                    symbol,
                    tick_time,
                    decision,
                    bid,
                    ask,
                    spread,
                    mid,
                    rsi_14,
                    atr_14,
                    liq_level,
                    liq_type,
                    liq_side,
                    liq_score,
                    liq_confirmed,
                    liq_swept,
                    dist_to_nearest_high,
                    dist_to_nearest_low,
                    session,
                    price_position,
                    bar_close
                ) VALUES (
                    $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,
                    $11,$12,$13,$14,$15,$16,$17,$18,$19,$20
                )
                ON CONFLICT (symbol, tick_time) DO NOTHING
            """,
                row["symbol"],
                row["timestamp_utc"],
                decision_str,
                ctx.bid,
                ctx.ask,
                ctx.spread,
                ctx.mid,
                ctx.rsi_14,
                ctx.atr_14,
                ctx.liq_level,
                ctx.liq_type,
                ctx.liq_side,
                ctx.liq_score,
                ctx.liq_confirmed,
                ctx.liq_swept,
                ctx.dist_to_nearest_high,
                ctx.dist_to_nearest_low,
                ctx.session,
                ctx.price_position,
                ctx.bar_close,
            )

            saved += 1
            logger.info(
                f"{decision_str:10s} | "
                f"BID={ctx.bid:.5f} | "
                f"RSI={ctx.rsi_14:.1f} | "
                f"Score={ctx.liq_score:.1f} | "
                f"Session={ctx.session} | "
                f"Position={ctx.price_position} | "
                f"Side={ctx.liq_side}"
            )

        logger.info(
            f"Gold layer complete — "
            f"{saved} decisions saved, {skipped} HOLDs skipped"
        )

    except Exception as exc:
        logger.error(f"Gold layer runner error: {exc}", exc_info=True)
        raise
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(run_gold_layer())