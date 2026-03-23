"""
Gold Layer — FeatureEngineer
Resamples unified ticks to 5-minute OHLC candles, computes RSI and ATR
using pandas-ta, identifies Liquidity Levels (swing highs/lows + round numbers),
then merges all features back to tick-level granularity via merge_asof.

The enriched DataFrame is persisted to DuckDBStore.tick_features.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import ta
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

if TYPE_CHECKING:
    from src.gold.duckdb_store import DuckDBStore

logger = logging.getLogger(__name__)

# ── Configuration constants ──────────────────────────────────────────────────
RESAMPLE_FREQ   = "5min"
RSI_PERIOD      = 14
ATR_PERIOD      = 14
SWING_WINDOW    = 5     # bars each side to confirm a swing high/low
ROUND_NUMBER_STEP = 10  # XAUUSD: every $10 is a round-number level (e.g. 2300, 2310)


class FeatureEngineer:
    """
    Computes backtest-ready features from a DataFrame of UnifiedTick rows.

    Usage:
        fe = FeatureEngineer()
        enriched_df = fe.build_features(unified_ticks_df)
        fe.save_to_duckdb(enriched_df, store)
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def build_features(self, ticks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Full feature pipeline:
          1. Resample ticks → 5-min OHLC
          2. Compute RSI(14) + ATR(14) on candles
          3. Identify Liquidity Levels
          4. Merge everything back to tick granularity

        Args:
            ticks_df: DataFrame with columns [timestamp_utc, bid, ask, volume, symbol, source]

        Returns:
            Tick-level DataFrame enriched with bar_open/high/low/close,
            rsi_14, atr_14, liq_level, liq_type columns.
        """
        if ticks_df.empty:
            logger.warning("[FeatureEngineer] Empty DataFrame received, skipping")
            return ticks_df

        ticks_df = ticks_df.copy()
        ticks_df["timestamp_utc"] = pd.to_datetime(ticks_df["timestamp_utc"], utc=True)
        ticks_df = ticks_df.sort_values("timestamp_utc")
        ticks_df["mid"] = (ticks_df["bid"] + ticks_df["ask"]) / 2.0

        # Step 1 — resample to OHLC candles
        candles = self._resample_ohlc(ticks_df)

        # Step 2 — compute RSI + ATR
        candles = self._compute_indicators(candles)

        # Step 3 — identify liquidity levels from candles
        liq_df = self._identify_liquidity_levels(candles)

        # Step 4 — merge back to tick resolution
        enriched = self._merge_to_ticks(ticks_df, candles, liq_df)

        logger.info(
            f"[FeatureEngineer] Built features: {len(enriched)} ticks, "
            f"{len(candles)} 5-min candles"
        )
        return enriched

    def save_to_duckdb(self, df: pd.DataFrame, store: "DuckDBStore") -> None:
        """Persist enriched tick DataFrame to the Gold layer DuckDB store."""
        if df.empty:
            logger.warning("[FeatureEngineer] Nothing to save — DataFrame is empty")
            return
        cols = [
            "timestamp_utc", "symbol", "bid", "ask", "volume", "source",
            "bar_open", "bar_high", "bar_low", "bar_close",
            "rsi_14", "atr_14", "liq_level", "liq_type",
        ]
        # Only keep columns that exist (liq cols may not always be present)
        available = [c for c in cols if c in df.columns]
        store.upsert_features(df[available])

    # ── Private helpers ───────────────────────────────────────────────────────

    def _resample_ohlc(self, ticks_df: pd.DataFrame) -> pd.DataFrame:
        """Resample mid-price to 5-min OHLC candles."""
        ohlc = (
            ticks_df.set_index("timestamp_utc")["mid"]
            .resample(RESAMPLE_FREQ)
            .ohlc()
            .dropna()
        )
        # Also aggregate volume per bar
        vol = (
            ticks_df.set_index("timestamp_utc")["volume"]
            .resample(RESAMPLE_FREQ)
            .sum()
        )
        candles = ohlc.join(vol, how="left")
        candles.index.name = "bar_time"
        candles.columns = ["bar_open", "bar_high", "bar_low", "bar_close", "bar_volume"]
        candles = candles.reset_index()
        # Ensure bar_time is timezone-aware UTC to match ticks_df
        candles["bar_time"] = pd.to_datetime(candles["bar_time"], utc=True)
        return candles

    def _compute_indicators(self, candles: pd.DataFrame) -> pd.DataFrame:
        """
        Compute RSI(14) and ATR(14) using the 'ta' library.
        RSI operates on close prices; ATR requires high/low/close.
        """
        candles = candles.copy()

        # ta library expects specific Series for each indicator
        rsi_indicator = RSIIndicator(close=candles["bar_close"], window=RSI_PERIOD)
        atr_indicator = AverageTrueRange(
            high=candles["bar_high"], 
            low=candles["bar_low"], 
            close=candles["bar_close"], 
            window=ATR_PERIOD
        )

        candles[f"rsi_{RSI_PERIOD}"] = rsi_indicator.rsi()
        candles[f"atr_{ATR_PERIOD}"] = atr_indicator.average_true_range()
        return candles

    def _identify_liquidity_levels(self, candles: pd.DataFrame) -> pd.DataFrame:
        """
        Identify three types of liquidity levels:
          1. Swing Highs — local maxima in `bar_high` (SWING_WINDOW bars each side)
          2. Swing Lows  — local minima in `bar_low`
          3. Round Numbers — multiples of ROUND_NUMBER_STEP within the price range

        Returns a DataFrame with columns [bar_time, liq_level, liq_type].
        """
        records: list[dict] = []
        highs = candles["bar_high"].values
        lows  = candles["bar_low"].values
        times = candles["bar_time"].values
        w = SWING_WINDOW

        for i in range(w, len(candles) - w):
            # Swing high
            if highs[i] == max(highs[i - w: i + w + 1]):
                records.append({"bar_time": times[i], "liq_level": highs[i], "liq_type": "swing_high"})
            # Swing low
            if lows[i] == min(lows[i - w: i + w + 1]):
                records.append({"bar_time": times[i], "liq_level": lows[i], "liq_type": "swing_low"})

        # Round-number levels in the observed price range
        price_min = candles["bar_low"].min()
        price_max = candles["bar_high"].max()
        step = ROUND_NUMBER_STEP
        first_level = (int(price_min / step) + 1) * step
        for level in np.arange(first_level, price_max, step):
            records.append({
                "bar_time": candles["bar_time"].iloc[0],  # static level — assign to start
                "liq_level": round(float(level), 2),
                "liq_type": "round_number",
            })

        if not records:
            return pd.DataFrame(columns=["bar_time", "liq_level", "liq_type"])
        
        liq_df = pd.DataFrame(records)
        liq_df["bar_time"] = pd.to_datetime(liq_df["bar_time"], utc=True)
        return liq_df

    def _merge_to_ticks(
        self,
        ticks_df: pd.DataFrame,
        candles: pd.DataFrame,
        liq_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge candle-level features back to tick resolution using merge_asof
        (each tick inherits from its most recent 5-min candle).
        """
        candles_sorted = candles.sort_values("bar_time")
        ticks_sorted   = ticks_df.sort_values("timestamp_utc")

        # Align candle fields onto each tick
        enriched = pd.merge_asof(
            ticks_sorted,
            candles_sorted[[
                "bar_time", "bar_open", "bar_high", "bar_low", "bar_close",
                f"rsi_{RSI_PERIOD}", f"atr_{ATR_PERIOD}"
            ]],
            left_on="timestamp_utc",
            right_on="bar_time",
            direction="backward",
        )
        enriched = enriched.rename(columns={
            f"rsi_{RSI_PERIOD}": "rsi_14",
            f"atr_{ATR_PERIOD}": "atr_14",
        })

        # Attach nearest liquidity level to each tick
        if not liq_df.empty:
            liq_sorted = liq_df.sort_values("bar_time")
            enriched = pd.merge_asof(
                enriched,
                liq_sorted[["bar_time", "liq_level", "liq_type"]],
                left_on="timestamp_utc",
                right_on="bar_time",
                direction="backward",
                suffixes=("", "_liq"),
            )
        else:
            enriched["liq_level"] = None
            enriched["liq_type"]  = None

        enriched.drop(columns=["mid", "bar_time", "bar_time_liq"], errors="ignore", inplace=True)
        return enriched
