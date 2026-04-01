import asyncio
import logging
import pandas as pd
from dotenv import load_dotenv

from src.validation.silver_processor import SilverProcessor
from src.gold.duckdb_store import DuckDBStore
from src.gold.feature_engineer import FeatureEngineer
from src.bot.audit_logger import run_gold_layer

# ── Configuration ─────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GOLD_DB_PATH = "data/gold/goldstream.duckdb"
PROCESS_INTERVAL_TICKS = 25  # Run Silver/Gold layers every N ticks

async def run():

    logger.info("GoldStream Live Pipeline (Scout & Sniper) running...")
    
    # Initialize components
    processor = SilverProcessor()
    fe = FeatureEngineer()
    
    # Ensure DuckDB schema is initialized
    with DuckDBStore(db_path=GOLD_DB_PATH) as store:
        store.init_schema()

    tick_count = 0
    buffer = []

    try:
        while True:
            raw_tick = fetch_tick()
            if raw_tick:
                # ── SILVER: Convert to UnifiedTick ────────────────────────────
                unified = processor.process_mt5_tick(raw_tick)
                if unified:
                    buffer.append(unified)
                    tick_count += 1
                    
                    # Log progress (quieter)
                    if tick_count % 5 == 0:
                        logger.info(f"Collected {tick_count} ticks...")

                    # ── PERIODIC GOLD PROCESSING ──────────────────────────────────
                    if tick_count % PROCESS_INTERVAL_TICKS == 0:
                        logger.info(f"--- Processing Batch (Tick {tick_count}) ---")
                        
                        with DuckDBStore(db_path=GOLD_DB_PATH) as store:
                            # 1. Save buffer to DuckDB
                            store.insert_unified_ticks(buffer)
                            buffer.clear()
                            
                            # 2. Extract recent history for feature calculation
                            # (We need enough candles for RSI/ATR windows)
                            # For simplicity here, we take everything for the current symbol.
                            # In production, you'd limit this to the last N bars.
                            query = f"SELECT * FROM unified_ticks WHERE symbol = '{unified.symbol}' ORDER BY timestamp_utc ASC"
                            ticks_df = store._con.execute(query).df()
                            
                            # 3. Build Features (BOS, CHoCH, FVG)
                            logger.info("[Gold] Computing SMC features...")
                            enriched_df = fe.build_features(ticks_df)
                            
                            if not enriched_df.empty:
                                # 4. Save Features
                                fe.save_to_duckdb(enriched_df, store)
                                logger.info(f"[Gold] Updated {len(enriched_df)} ticks with structure & FVG data")
                        
                        # 5. Run Strategy (Audit Logger)
                        # We run this outside the store context to allow audit_logger to open its own connection
                        logger.info("[Bot] Running Scout & Sniper decision engine...")
                        await run_gold_layer(db_path=GOLD_DB_PATH)

            await asyncio.sleep(0.5)  # Fast-poll MT5 ticks

    except KeyboardInterrupt:
        logger.info("Pipeline stopped by user")
    except Exception as exc:
        logger.error(f"Pipeline error: {exc}", exc_info=True)
    finally:
        logger.info("Shutting down...")

if __name__ == "__main__":
    asyncio.run(run())