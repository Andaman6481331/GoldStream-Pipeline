"""
Example Backtest — Gold RSI & ATR Mean Reversion
Uses the DuckDB Gold layer to simulate a simple strategy.

Strategy:
  1. LONG  when RSI < 30 (oversold)
  2. SHORT when RSI > 70 (overbought)
  3. Uses a Trailing Stop based on ATR (Volatility-adjusted)
  4. Spread-aware: liquidity gaps trigger force-closes
"""

from datetime import datetime, timezone
from src.gold.duckdb_store import DuckDBStore
from src.backtest.backtest_engine import BacktestEngine, Action, Position, TickEvent
import logging

# Configure logging to see trade details
logging.basicConfig(level=logging.INFO, format="%(message)s")

def mean_reversion_strategy(event: TickEvent) -> Action:
    """
    Simple RSI-based strategy.
    
    event.rsi_14: Computed on 5-min candles
    event.atr_14: Volatility measurement
    """
    # If no position, look for entry
    if event.current_position == Position.FLAT:
        if event.rsi_14 and event.rsi_14 < 30:
            return Action.OPEN_LONG
        if event.rsi_14 and event.rsi_14 > 70:
            return Action.OPEN_SHORT
            
    # If in LONG, look for exit (overbought)
    elif event.current_position == Position.LONG:
        if event.rsi_14 and event.rsi_14 > 65:
            return Action.CLOSE
            
    # If in SHORT, look for exit (oversold)
    elif event.current_position == Position.SHORT:
        if event.rsi_14 and event.rsi_14 < 35:
            return Action.CLOSE
            
    return Action.HOLD

def run_it():
    # 1. Connect to the Gold layer
    with DuckDBStore() as store:
        # 2. Setup the engine
        # XAUUSD pips are $0.01. We use a 100-pip ($1.00) trailing stop.
        engine = BacktestEngine(
            store, 
            symbol="XAUUSD", 
            trailing_stop_pips=100.0,  # $1.00 move
            gap_threshold_pips=30.0    # 30 cent spread spike = liquidity gap
        )
        
        # 3. Run search over the date range we just ingested
        print("Starting backtest...")
        result = engine.run(
            strategy_fn=mean_reversion_strategy,
            from_dt=datetime(2024, 1, 1, tzinfo=timezone.utc),
            to_dt=datetime(2024, 1, 7, tzinfo=timezone.utc)
        )
        
        # 4. Show results
        print(result)
        
        if result.trades:
            print("\n-- Last 5 trades --")
            for t in result.trades[-5:]:
                color = "+" if t.pnl > 0 else "-"
                print(f"[{t.exit_time}] {t.direction.value} | PnL: {color}{abs(t.pnl):.2f} | Reason: {t.exit_reason}")

if __name__ == "__main__":
    run_it()
