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
import webbrowser
import tempfile
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
    con = duckdb.connect(db_path, read_only=True)
    
    # 1. Query Primary Candles (15m default)
    candle_table = "candles_15m" if timeframe == "15min" else "candles_1m"
    candle_query = f"""
        SELECT 
            bar_time as timestamp_utc, 
            bar_open, bar_high, bar_low, bar_close, 
            smc_trend_15m, 
            bos_detected_15m, choch_detected_15m,
            bos_up_15m, bos_down_15m, choch_up_15m, choch_down_15m,
            is_swing_high_15m, is_swing_low_15m,
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

    # 4. Add Trace: BOS/CHoCH and Swing Markers
    # Swing Highs
    sh = df_candles[df_candles['is_swing_high_15m'] == True]
    if not sh.empty:
        fig.add_trace(go.Scatter(
            x=sh['timestamp_utc'], y=sh['bar_high'] + 1.0,
            mode='markers', marker=dict(symbol='triangle-down', size=8, color='rgba(239, 83, 80, 0.8)'),
            name="Swing High"
        ))
        
    # Swing Lows
    sl = df_candles[df_candles['is_swing_low_15m'] == True]
    if not sl.empty:
        fig.add_trace(go.Scatter(
            x=sl['timestamp_utc'], y=sl['bar_low'] - 1.0,
            mode='markers', marker=dict(symbol='triangle-up', size=8, color='rgba(38, 166, 154, 0.8)'),
            name="Swing Low"
        ))

    # BOS Up
    bos_up = df_candles[df_candles['bos_up_15m'] == True]
    if not bos_up.empty:
        fig.add_trace(go.Scatter(
            x=bos_up['timestamp_utc'], y=bos_up['bar_high'] + 2.0,
            mode='text', text="BOS ↑", textposition="top center",
            textfont=dict(color='rgba(38, 166, 154, 1)', size=10, family="Arial Black"),
            name="BOS Up"
        ))

    # BOS Down
    bos_down = df_candles[df_candles['bos_down_15m'] == True]
    if not bos_down.empty:
        fig.add_trace(go.Scatter(
            x=bos_down['timestamp_utc'], y=bos_down['bar_low'] - 2.0,
            mode='text', text="BOS ↓", textposition="bottom center",
            textfont=dict(color='rgba(239, 83, 80, 1)', size=10, family="Arial Black"),
            name="BOS Down"
        ))

    # CHoCH Up
    choch_up = df_candles[df_candles['choch_up_15m'] == True]
    if not choch_up.empty:
        fig.add_trace(go.Scatter(
            x=choch_up['timestamp_utc'], y=choch_up['bar_low'] - 2.5,
            mode='markers+text', text="CHoCH ↑", textposition="bottom center",
            marker=dict(symbol='star', size=12, color='rgba(38, 166, 154, 1)'),
            textfont=dict(color='rgba(38, 166, 154, 1)', size=11, family="Arial Black"),
            name="CHoCH Up"
        ))

    # CHoCH Down
    choch_down = df_candles[df_candles['choch_down_15m'] == True]
    if not choch_down.empty:
        fig.add_trace(go.Scatter(
            x=choch_down['timestamp_utc'], y=choch_down['bar_high'] + 2.5,
            mode='markers+text', text="CHoCH ↓", textposition="top center",
            marker=dict(symbol='star', size=12, color='rgba(239, 83, 80, 1)'),
            textfont=dict(color='rgba(239, 83, 80, 1)', size=11, family="Arial Black"),
            name="CHoCH Down"
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
            
    # 5b. Highlight Background Trend Zones
    # Identify contiguous blocks of "bull" or "bear" in smc_trend_15m
    if 'smc_trend_15m' in df_candles.columns:
        current_trend = None
        trend_start = None
        dfc = df_candles.dropna(subset=['smc_trend_15m', 'timestamp_utc'])
        for i, row in dfc.iterrows():
            trend = row['smc_trend_15m']
            t_time = row['timestamp_utc']
            
            # Trend changed or starting
            if trend != current_trend:
                if current_trend in ['bull', 'bear']:
                    # Close previous rect
                    color = 'rgba(38, 166, 154, 0.1)' if current_trend == 'bull' else 'rgba(239, 83, 80, 0.1)'
                    fig.add_vrect(
                        x0=trend_start, x1=t_time,
                        fillcolor=color, opacity=1,
                        layer="below", line_width=0,
                        name=f"Trend: {current_trend.upper()}"
                    )
                current_trend = trend
                trend_start = t_time
                
        # Close the last open rect
        if current_trend in ['bull', 'bear'] and trend_start is not None:
            color = 'rgba(38, 166, 154, 0.1)' if current_trend == 'bull' else 'rgba(239, 83, 80, 0.1)'
            fig.add_vrect(
                x0=trend_start, x1=dfc.iloc[-1]['timestamp_utc'],
                fillcolor=color, opacity=1,
                layer="below", line_width=0,
                name=f"Trend: {current_trend.upper()}"
            )

    # 6. Formatting
    fig.update_layout(
        title=f"SMC Audit: {symbol} | 4H Bias: {market_bias.upper()}",
        template="plotly_dark",
        xaxis_rangeslider_visible=True,
        yaxis_title="Price",
        xaxis_title="Time (UTC)",
        yaxis=dict(
            autorange=True, 
            fixedrange=False,
            tickmode='linear',
            dtick=10, # Default interval
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)'
        ),
        margin=dict(l=50, r=50, t=80, b=50)
    )

    # 7. Add Interactive Scroll Logic via JS Injection
    # We use fig.to_html and inject a small JS script to handle the wheel event
    # for changing the price level interval (dtick).
    config = {'scrollZoom': True, 'responsive': True}
    html_content = fig.to_html(config=config, include_plotlyjs='cdn', full_html=True)
    
    js_interactive = """
    <script>
    window.addEventListener('load', function() {
        const plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
        const intervals = [0.5, 1, 2, 5, 10, 20, 25, 50, 100, 200, 500, 1000];
        let currentIdx = 4; // Start at 10 (intervals[4])

        // HUD for feedback
        const hud = document.createElement('div');
        hud.id = 'price-step-hud';
        hud.style.position = 'fixed';
        hud.style.top = '20px';
        hud.style.right = '20px';
        hud.style.background = 'rgba(38, 166, 154, 0.8)';
        hud.style.color = 'white';
        hud.style.padding = '8px 15px';
        hud.style.borderRadius = '4px';
        hud.style.fontFamily = 'monospace';
        hud.style.fontSize = '14px';
        hud.style.zIndex = '9999';
        hud.style.pointerEvents = 'none';
        hud.style.boxShadow = '0 2px 10px rgba(0,0,0,0.5)';
        hud.innerHTML = 'STEP: 10 USD';
        document.body.appendChild(hud);

        function updateStep(delta) {
            if (delta > 0) {
                // Scroll down: decrease interval (more lines)
                currentIdx = Math.max(0, currentIdx - 1);
            } else {
                // Scroll up: increase interval (less lines)
                currentIdx = Math.min(intervals.length - 1, currentIdx + 1);
            }
            const newDtick = intervals[currentIdx];
            Plotly.relayout(plotDiv, {'yaxis.dtick': newDtick});
            hud.innerHTML = 'STEP: ' + newDtick + ' USD';
            hud.style.background = 'rgba(239, 83, 80, 0.8)'; // Feedback color change
            setTimeout(() => { hud.style.background = 'rgba(38, 166, 154, 0.8)'; }, 200);
        }

        plotDiv.on('wheel', function(e) {
            // If they are holding Ctrl, let native zoom work
            if (e.ctrlKey) return;
            
            // Otherwise, adjust price levels
            e.preventDefault();
            updateStep(e.deltaY);
        }, {passive: false});
    });
    </script>
    """
    
    html_content = html_content.replace('</body>', js_interactive + '</body>')

    logger.info("Opening interactive audit in browser...")
    
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as f:
        f.write(html_content)
        temp_path = f.name
    
    webbrowser.open('file://' + temp_path)

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
