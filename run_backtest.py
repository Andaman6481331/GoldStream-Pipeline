"""
Scout & Sniper Standalone Backtest Launcher

Executes the full event-based BacktestEngine over the DuckDB Gold Layer.
Tracks exact Entry, Exit, SL, TP, and Unrealized/Realized PnL.
"""

import argparse
import logging
import sys
from datetime import datetime
import pandas as pd

from src.gold.duckdb_store import DuckDBStore
from src.backtest.backtest_engine import BacktestEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("run_backtest")

def parse_args():
    parser = argparse.ArgumentParser(description="Run full backtest lifecycle over Gold dataset.")
    parser.add_argument("--symbol", default="XAUUSD", help="Symbol to test (default XAUUSD)")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--db", default="data/gold/goldstream.duckdb", help="Path to DuckDB")
    
    # Strategy settings exposed through CLI for easy tweaking
    parser.add_argument("--ts-pips", type=float, default=50.0, help="Fixed trailing stop fallback pips if ATR is unavailable")
    parser.add_argument("--atr-mult", type=float, default=1.5, help="Multiplier for ATR-based dynamic stop loss")
    parser.add_argument("--spread-limit", type=float, default=30.0, help="Maximum allowable spread in pips before force close")
    return parser.parse_args()

def main():
    args = parse_args()

    # Parse dates safely to UTC
    try:
        start_dt = pd.to_datetime(args.start).tz_localize("UTC").to_pydatetime()
        end_dt   = pd.to_datetime(args.end).tz_localize("UTC").to_pydatetime()
    except Exception as e:
        logger.error(f"Failed to parse dates. Ensure format is YYYY-MM-DD: {e}")
        sys.exit(1)

    logger.info(f"Connecting to DuckDB: {args.db}")
    try:
        with DuckDBStore(db_path=args.db, read_only=True) as store:
            logger.info(f"Initialising Backtest Engine for {args.symbol} from {args.start} to {args.end}")
            logger.info(f"Params: TS_PIPS={args.ts_pips}, ATR_MULT={args.atr_mult}, MAX_SPREAD={args.spread_limit}")
            
            engine = BacktestEngine(
                store=store,
                symbol=args.symbol,
                trailing_stop_pips=args.ts_pips,
                wide_spread_pips=args.spread_limit,
                atr_stop_multiplier=args.atr_mult
            )

            # Execution
            logger.info("Executing tick-by-tick simulation (this may take a moment)...")
            result = engine.run(from_dt=start_dt, to_dt=end_dt)

            # The result object's __str__ now prints the categorical summaries
            print(result)
            
            # Export CSV of trades for external review if there are any
            if result.trades:
                trades_dict = []
                for t in result.trades:
                    t_dict = t.__dict__.copy()
                    # Convert Enums to strings for CSV compatibility
                    t_dict['direction'] = t.direction.value if hasattr(t.direction, 'value') else str(t.direction)
                    trades_dict.append(t_dict)
                    
                export_file = f"backtest_{args.symbol}_{args.start}_to_{args.end}.csv"
                pd.DataFrame(trades_dict).to_csv(export_file, index=False)
                logger.info(f"Exported {len(result.trades)} completed trades to '{export_file}'")
                
    except FileNotFoundError:
        logger.error(f"Database not found at {args.db}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Simulation failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
