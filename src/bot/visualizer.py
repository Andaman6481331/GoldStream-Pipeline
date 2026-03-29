"""
Scout & Sniper Visualizer
Stand-alone audit tool to view SMC features (BOS/CHoCH/Bias) 
directly from the Gold DuckDB.
"""

import plotly.graph_objects as go
import duckdb
import pandas as pd
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("visualizer")

def run_visualizer(db_path="data/gold/goldstream.duckdb", symbol="XAUUSD", timeframe="15min"):
    """
    Query candle tables for OHLC/SMC and tick_features for overlays (FVGs).
    """
    if not Path(db_path).exists():
        logger.error(f"Database not found: {db_path}")
        return

    logger.info(f"Connecting to {db_path} for {symbol} ({timeframe})...")
    con = duckdb.connect(db_path)
    
    # 1. Query Primary Candles (15m default)
    candle_table = "candles_15m" if timeframe == "15min" else "candles_1m"
    candle_query = f"""
        SELECT 
            bar_time as timestamp_utc, 
            bar_open, bar_high, bar_low, bar_close, 
            smc_trend_15m, 
            bos_detected_15m, choch_detected_15m,
            hh_15m, ll_15m
        FROM {candle_table}
        WHERE symbol = '{symbol}'
        ORDER BY bar_time ASC
    """
    df_candles = con.execute(candle_query).df()

    # 4H Bias comes from a separate table usually
    bias_query = f"SELECT bar_time, market_bias_4h FROM candles_4h WHERE symbol = '{symbol}' ORDER BY bar_time DESC LIMIT 1"
    bias_res = con.execute(bias_query).fetchone()
    market_bias = bias_res[1] if bias_res else "neutral"

    # 2. Query Features Overlay (FVGs) from tick_features
    # We filter for rows where FVG is present to keep it light
    fvg_query = f"""
        SELECT 
            timestamp_utc, 
            fvg_high, fvg_low, fvg_side, fvg_filled
        FROM tick_features 
        WHERE symbol = '{symbol}' 
          AND fvg_high IS NOT NULL
        ORDER BY timestamp_utc ASC
    """
    df_features = con.execute(fvg_query).df()
    con.close()

    if df_candles.empty:
        logger.warning(f"No data found in {candle_table}. Run the pipeline first!")
        return

    # 3. Create Trace: Candlesticks
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df_candles['timestamp_utc'],
        open=df_candles['bar_open'],
        high=df_candles['bar_high'],
        low=df_candles['bar_low'],
        close=df_candles['bar_close'],
        name="Price Action",
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ))

    # 4. Add Trace: BOS/CHoCH Markers
    bos = df_candles[df_candles['bos_detected_15m'] == True]
    if not bos.empty:
        fig.add_trace(go.Scatter(
            x=bos['timestamp_utc'],
            y=bos['bar_high'],
            mode='markers',
            marker=dict(symbol='triangle-up', size=12, color='royalblue'),
            name="BOS (Break)"
        ))

    choch = df_candles[df_candles['choch_detected_15m'] == True]
    if not choch.empty:
        fig.add_trace(go.Scatter(
            x=choch['timestamp_utc'],
            y=choch['bar_high'],
            mode='markers',
            marker=dict(symbol='star', size=14, color='darkorchid'),
            name="CHoCH (Reversal)"
        ))

    # 5. Add Trace: FVG Zones (Visualized as boxes or lines)
    if not df_features.empty:
        # Drawing only unique FVGs to avoid over-drawing
        for _, fvg in df_features.drop_duplicates(subset=['fvg_high', 'fvg_low']).iterrows():
            color = 'rgba(38, 166, 154, 0.2)' if fvg['fvg_side'] == 'bullish_fvg' else 'rgba(239, 83, 80, 0.2)'
            fig.add_hrect(
                y0=fvg['fvg_low'], y1=fvg['fvg_high'],
                fillcolor=color, opacity=0.5,
                layer="below", line_width=0,
                name=f"FVG {fvg['fvg_side']}"
            )

    # 6. Formatting
    fig.update_layout(
        title=f"SMC Audit: {symbol} | 4H Bias: {market_bias.upper()}",
        template="plotly_dark",
        xaxis_rangeslider_visible=True, # Added slider for easier navigation
        yaxis_title="Price",
        xaxis_title="Time (UTC)",
        yaxis=dict(autorange=True, fixedrange=False) # Auto-fit the price axis
    )

    logger.info("Opening browser for interactive audit...")
    fig.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",      default="data/gold/goldstream.duckdb")
    parser.add_argument("--symbol",  default="XAUUSD")
    parser.add_argument("--tf",      default="15min")
    args = parser.parse_args()
    
    # Check if database exists relative to CWD; if not, try root-level data/gold...
    db_path = args.db
    if not Path(db_path).exists():
        # Try one level up if we're in src/bot/
        if Path(f"../../{args.db}").exists():
            db_path = f"../../{args.db}"
        else:
            logger.error(f"Cannot find database at {args.db}. Please run from the project root using: python -m src.bot.visualizer")
            sys.exit(1)
            
    run_visualizer(db_path=db_path, symbol=args.symbol, timeframe=args.tf)
