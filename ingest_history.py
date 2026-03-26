"""
Historical Ingestion Entry Point
Orchestrates the full Bronze → Silver → Gold pipeline for Dukascopy historical data.

Usage:
    python ingest_history.py --symbol XAUUSD --start 2024-01-01 --end 2024-01-31

    # Full 5-year run (will take several minutes)
    python ingest_history.py --symbol XAUUSD --start 2020-01-01 --end 2024-12-31

Environment:
    No .env file required — this pipeline is fully file-based (Parquet + DuckDB).
    Set LOG_LEVEL=DEBUG for verbose output.
"""

import argparse
import asyncio
import logging
import sys
import os
from pathlib import Path

# ── Ensure project root is on sys.path ───────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ingestion.history_downloader import HistoryDownloader
from src.validation.silver_processor  import SilverProcessor
from src.gold.duckdb_store            import DuckDBStore
from src.gold.feature_engineer        import FeatureEngineer
from src.bot.audit_logger             import run_gold_layer

import pandas as pd

# ── Logging setup ─────────────────────────────────────────────────────────────
log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ingest_history")


# ── Pipeline ──────────────────────────────────────────────────────────────────

async def run_pipeline(
    symbol: str,
    start: str,
    end: str,
    bronze_dir: str = "data/bronze",
    gold_db:    str = "data/gold/goldstream.duckdb",
    skip_download: bool = False,
) -> None:
    """
    Full historical ingestion pipeline:
      1. Bronze  — Download .bi5 files → Parquet partitions
      2. Silver  — Validate + normalise to UnifiedTick via Pydantic
      3. Gold    — Compute RSI/ATR/Liquidity features, store to DuckDB
    """

    # ── BRONZE ────────────────────────────────────────────────────────────────
    if not skip_download:
        logger.info(f"[Bronze] Starting download: {symbol} {start} → {end}")
        downloader = HistoryDownloader(
            symbol=symbol,
            output_dir=bronze_dir,
            max_concurrent=8,
        )
        summary = await downloader.download_range(start, end)
        logger.info(
            f"[Bronze] Complete — {summary['ticks_saved']:,} ticks saved, "
            f"{summary['files_written']} Parquet files written"
        )
    else:
        logger.info("[Bronze] Skipping download (--skip-download flag set)")

    # ── SILVER ────────────────────────────────────────────────────────────────
    logger.info("[Silver] Processing Parquet partitions → UnifiedTick")
    processor = SilverProcessor()
    unified_ticks = list(processor.process_all_parquets(bronze_dir, symbol=symbol))
    logger.info(f"[Silver] Validated {len(unified_ticks):,} UnifiedTick rows")

    if not unified_ticks:
        logger.error("[Silver] No ticks produced — check bronze data and logs. Aborting.")
        return

    # ── GOLD (DuckDB) ─────────────────────────────────────────────────────────
    logger.info("[Gold] Initialising DuckDB feature store")
    gold_path = Path(gold_db)
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    
    with DuckDBStore(db_path=gold_db) as store:
        store.init_schema()

        # Insert unified ticks
        logger.info("[Gold] Inserting UnifiedTick rows into DuckDB")
        inserted = store.insert_unified_ticks(iter(unified_ticks))
        logger.info(f"[Gold] {inserted:,} rows in unified_ticks")

        # Build features
        logger.info("[Gold] Computing RSI / ATR / Liquidity features")
        ticks_df = pd.DataFrame([t.model_dump() for t in unified_ticks])
        ticks_df["timestamp_utc"] = pd.to_datetime(ticks_df["timestamp_utc"], utc=True)

        fe = FeatureEngineer()
        enriched = fe.build_features(ticks_df)

        if not enriched.empty:
            fe.save_to_duckdb(enriched, store)
            logger.info(
                f"[Gold] Features saved — {len(enriched):,} rows in tick_features"
            )

            # ── STRATEGY EXECUTION (Scout & Sniper) ───────────────────────
            logger.info("[Gold] Running Scout & Sniper strategy engine...")
            # We close the store context first to ensure everything is flushed, 
            # or we just call the runner within the same process if it handles its own connection.
            # audit_logger.run_gold_layer() creates its own DuckDBStore connection.

            # Print a quick sample
            sample_cols = ["timestamp_utc", "bid", "ask", "rsi_14", "atr_14", "liq_level", "liq_type"]
            available   = [c for c in sample_cols if c in enriched.columns]
            non_null    = enriched[enriched["rsi_14"].notna()]
            if not non_null.empty:
                print("\n-- Sample of enriched ticks (first 5 with RSI) --")
                print(non_null[available].head().to_string(index=False))
        else:
            logger.warning("[Gold] Feature engineering returned an empty DataFrame")

        # Final counts
        uc = store.get_tick_count("unified_ticks")
        fc = store.get_tick_count("tick_features")
        print(f"\n-- DuckDB totals: unified_ticks={uc:,}  tick_features={fc:,} --")

    # Run strategy after store is closed to avoid lock issues
    await run_gold_layer(db_path=gold_db)

    logger.info("[Pipeline] Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GoldStream Historical Ingestion — Dukascopy → Parquet → DuckDB"
    )
    parser.add_argument("--symbol",        default="XAUUSD", help="Dukascopy symbol (default: XAUUSD)")
    parser.add_argument("--start",         required=True,    help="Start date YYYY-MM-DD")
    parser.add_argument("--end",           required=True,    help="End date YYYY-MM-DD (inclusive)")
    parser.add_argument("--bronze-dir",    default="data/bronze", help="Bronze Parquet output dir")
    parser.add_argument("--gold-db",       default="data/gold/goldstream.duckdb", help="DuckDB path")
    parser.add_argument("--skip-download", action="store_true",  help="Skip Bronze download, process existing Parquet")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_pipeline(
        symbol        = args.symbol,
        start         = args.start,
        end           = args.end,
        bronze_dir    = args.bronze_dir,
        gold_db       = args.gold_db,
        skip_download = args.skip_download,
    ))
