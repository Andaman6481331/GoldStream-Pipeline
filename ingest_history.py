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
from datetime import timedelta, datetime
from pathlib import Path

# ── Ensure project root is on sys.path ───────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ingestion.history_downloader import HistoryDownloader
from src.validation.silver_processor  import SilverProcessor
from src.gold.duckdb_store            import DuckDBStore
from src.gold.feature_engineer        import FeatureEngineer
from src.bot.audit_logger             import run_gold_layer
from src.backtest.backtest_engine     import BacktestEngine

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
    skip_silver:   bool = False,
    skip_gold:     bool = False,
    skip_backtest: bool = False,
    skip_audit:    bool = False,
    limit_audit:   int  = 1000,
    only_backtest: bool = False,
) -> None:
    """
    Full historical ingestion pipeline:
      1. Bronze  — Download .bi5 files → Parquet partitions
      2. Silver  — Validate + normalise to UnifiedTick via Pydantic
      3. Gold    — Compute RSI/ATR/Liquidity features, store to DuckDB
      4. Audit   — Match intent against features (quietly to audit.log)
      5. Backtest— High-fidelity simulation (to backtest.log)
    """

    # ── LOGGING SETUP (File-Based) ────────────────────────────────────────────
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure Backtest and Audit specific file handlers
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    
    audit_handler = logging.FileHandler(log_dir / "audit.log")
    audit_handler.setFormatter(formatter)
    logging.getLogger("src.bot.audit_logger").addHandler(audit_handler)
    
    backtest_handler = logging.FileHandler(log_dir / "backtest.log")
    backtest_handler.setFormatter(formatter)
    logging.getLogger("src.backtest.backtest_engine").addHandler(backtest_handler)

    if only_backtest:
        logger.info("[Pipeline] --only-backtest set: Skipping ingestion/feature/audit stages.")
        skip_download = skip_silver = skip_gold = skip_audit = True

    # Parse dates safely for range-bound tasks
    start_dt = pd.to_datetime(start).tz_localize("UTC").to_pydatetime()
    end_dt   = (pd.to_datetime(end).tz_localize("UTC") + timedelta(days=1) - timedelta(seconds=1)).to_pydatetime()

    # ── BRONZE ────────────────────────────────────────────────────────────────
    downloader = HistoryDownloader(
        symbol=symbol,
        output_dir=bronze_dir,
        max_concurrent=4,
    )

    if not skip_download:
        logger.info(f"[Bronze] Starting download: {symbol} {start} → {end}")
        summary = await downloader.download_range(start, end)
        logger.info(
            f"[Bronze] Complete — {summary['ticks_saved']:,} ticks saved, "
            f"{summary['files_written']} Parquet files written"
        )
    else:
        logger.info("[Bronze] Skipping download (--skip-download flag set)")

    # ── CONSOLIDATE ───────────────────────────────────────────────────────────
    # Since HistoryDownloader v2+ writes hourly files, we must merge them
    # into ticks.parquet for the SilverProcessor.
    logger.info("[Bronze] Consolidating hourly files → month partitions")
    months = pd.date_range(
        start=pd.Timestamp(start).replace(day=1),
        end=pd.Timestamp(end),
        freq="MS"
    )
    for m in months:
        downloader.merge_month(m.year, m.month)

    # ── SILVER ────────────────────────────────────────────────────────────────
    unified_ticks = []
    if not skip_silver:
        logger.info("[Silver] Processing Parquet partitions → UnifiedTick")
        processor = SilverProcessor()
        unified_ticks = list(processor.process_all_parquets(bronze_dir, symbol=symbol, start_date=start, end_date=end))
        logger.info(f"[Silver] Validated {len(unified_ticks):,} UnifiedTick rows")

        if not unified_ticks:
            logger.error("[Silver] No ticks produced — check bronze data and logs. Aborting.")
            return
    else:
        logger.info("[Silver] Skipping historical validation (--skip-silver or --only-backtest set)")

    # ── GOLD (DuckDB) ─────────────────────────────────────────────────────────
    logger.info("[Gold] Initialising DuckDB feature store")
    gold_path = Path(gold_db)
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    
    with DuckDBStore(db_path=gold_db) as store:
        store.init_schema()

        if not skip_silver:
            # Insert unified ticks
            logger.info("[Gold] Inserting UnifiedTick rows into DuckDB")
            inserted = store.insert_unified_ticks(iter(unified_ticks))
            logger.info(f"[Gold] {inserted:,} rows in unified_ticks")
            ticks_df = pd.DataFrame([t.model_dump() for t in unified_ticks])
        else:
            logger.info("[Gold] Skipping Silver — loading UnifiedTick rows from DuckDB store")
            ticks_df = store._con.execute("SELECT * FROM unified_ticks").df()

        # ── STRATEGY EXECUTION (Scout & Sniper) ───────────────────────
        if not skip_gold:
            # Build features
            logger.info("[Gold] Computing SMC features (BOS/FVG/Liquidity)")
            ticks_df["timestamp_utc"] = pd.to_datetime(ticks_df["timestamp_utc"], utc=True)

            fe = FeatureEngineer()
            enriched = fe.build_features(ticks_df)
            
            # TEMP DEBUG
            print("DEBUG bos_detected_15m:", enriched.get("bos_detected_15m", pd.Series()).sum())
            print("DEBUG sweep_candle_low non-null:", enriched["sweep_candle_low"].notna().sum() if "sweep_candle_low" in enriched.columns else "COLUMN MISSING")
            print("DEBUG sweep_candle_high non-null:", enriched["sweep_candle_high"].notna().sum() if "sweep_candle_high" in enriched.columns else "COLUMN MISSING")
            print("DEBUG fvg_high non-null:", enriched["fvg_high"].notna().sum() if "fvg_high" in enriched.columns else "COLUMN MISSING")
            print("DEBUG market_bias_4h counts:", enriched["market_bias_4h"].value_counts().to_dict() if "market_bias_4h" in enriched.columns else "COLUMN MISSING")

            if not enriched.empty:
                fe.save_to_duckdb(enriched, store)
                logger.info(
                    f"[Gold] SMC features saved — {len(enriched):,} rows in tick_features"
                )
            else:
                logger.warning("[Gold] Feature engineering returned an empty DataFrame")
        else:
            logger.info("[Gold] Skipping feature engineering (--skip-gold flag set)")

        logger.info("[Gold] Running Scout & Sniper strategy engine...")

    # Run strategy after store is closed
    if not skip_audit:
        logger.info(f"[Audit] Running intent engine ({start} → {end}) — Details in logs/audit.log")
        await run_gold_layer(
            db_path=gold_db,
            from_dt=start_dt,
            to_dt=end_dt,
            limit=limit_audit if limit_audit > 0 else None
        )
    else:
        logger.info("[Audit] Skipping intent engine (--skip-audit flag set)")

    # ── REPORTING & BACKTEST ──────────────────────────────────────────────────
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    if not skip_backtest:
        logger.info(f"[Backtest] Executing simulation: {symbol} {start} → {end} — Details in logs/backtest.log")
        
        with DuckDBStore(db_path=gold_db, read_only=True) as store:
            engine = BacktestEngine(
                store=store,
                symbol=symbol
            )
            result = engine.run(from_dt=start_dt, to_dt=end_dt)
            # Result print will go to both terminal and backtest.log
            
            if result.trades:
                trades_dict = []
                for t in result.trades:
                    t_dict = t.__dict__.copy()
                    # Convert Enums to strings for CSV compatibility
                    if hasattr(t.direction, 'value'):
                        t_dict['direction'] = t.direction.value
                    else:
                        t_dict['direction'] = str(t.direction)
                    trades_dict.append(t_dict)
                    
                export_file = reports_dir / f"backtest_{symbol}_{start}_to_{end}.csv"
                pd.DataFrame(trades_dict).to_csv(export_file, index=False)
                logger.info(f"[Backtest] Exported {len(result.trades)} trades to '{export_file}'")
    else:
        logger.info("[Backtest] Skipping simulation (--skip-backtest flag set)")

    # Export Decisions Log
    logger.info("[Report] Exporting decision audit log to CSV...")
    with DuckDBStore(db_path=gold_db, read_only=True) as store:
        decisions_df = store._con.execute(f"""
            SELECT * FROM trade_decisions 
            WHERE symbol = '{symbol}'
              AND decision != 'HOLD'
            ORDER BY tick_time ASC
        """).df()
        
        if not decisions_df.empty:
            decision_file = reports_dir / f"decisions_{symbol}_{start}_to_{end}.csv"
            decisions_df.to_csv(decision_file, index=False)
            logger.info(f"[Report] Exported {len(decisions_df)} trade decisions (excluding HOLDS) to '{decision_file}'")

    # ── FINAL SUMMARY PREVIEW ─────────────────────────────────────────────────
    if not skip_backtest:
        print("\n" + "="*50)
        print(f" PIPELINE SUMMARY: {symbol} ({start} to {end})")
        print("="*50)
        
        # Re-query briefly for comparison
        with DuckDBStore(db_path=gold_db, read_only=True) as store:
            stats = store._con.execute(f"""
                SELECT 
                    COUNT(*) FILTER (WHERE bos_detected_15m OR choch_detected_15m) as total_signals,
                    (SELECT COUNT(*) FROM trade_decisions WHERE symbol = '{symbol}' AND decision != 'HOLD') as audit_actions
                FROM tick_features
                WHERE symbol = '{symbol}'
                  AND timestamp_utc >= '{start_dt}' 
                  AND timestamp_utc <= '{end_dt}'
            """).fetchone() or (0, 0)
            
            sig_count, audit_count = stats
            trades_count = len(result.trades) if 'result' in locals() else 0
            
            print(f" SMC Signals (BOS/CHoCH) : {sig_count}")
            print(f" Audit Intent (non-HOLD) : {audit_count}")
            print(f" Backtest Executed Trades: {trades_count}")
            
            if sig_count > trades_count:
                print(f" ⚠️  Potential Missed Ops : {sig_count - trades_count}")
            
            if 'result' in locals():
                print("-" * 50)
                print(f" Net PnL: {result.gross_pnl:+.2f} | Win Rate: {result.win_rate:.1%}")
        print("="*50 + "\n")

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
    parser.add_argument("--skip-download", action="store_true",  help="Skip Bronze download")
    parser.add_argument("--skip-silver",   action="store_true",  help="Skip Silver processing (load from DuckDB)")
    parser.add_argument("--skip-gold",     action="store_true",  help="Skip Gold processing (load from DuckDB)")
    parser.add_argument("--skip-audit",    action="store_true",  help="Skip Audit intent phase")
    parser.add_argument("--skip-backtest", action="store_true",  help="Skip Backtest simulation")
    parser.add_argument("--only-backtest", action="store_true",  help="Run ONLY the backtest on existing DB data")
    parser.add_argument("--limit-audit",   type=int, default=0, help="Max ticks for Audit (0=unlimited, default)")

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
            skip_silver   = args.skip_silver,
            skip_gold     = args.skip_gold,
            skip_backtest = args.skip_backtest,
            skip_audit    = args.skip_audit,
            limit_audit   = args.limit_audit,
            only_backtest = args.only_backtest,
        ))
