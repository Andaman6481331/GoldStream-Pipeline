"""
Gold Layer — FeatureEngineer  (Scout & Sniper)
────────────────────
1.  BOS / CHoCH detection
        _compute_structure_breaks() — per-candle structural event labels
        Promotes the existing swing confirmation logic into named break events:
            bos_detected   (bool) — continuation break in trend direction
            choch_detected (bool) — reversal break against prior trend
            structure_direction (str) — current bias: "bullish" | "bearish"

2.  Fair Value Gap (FVG) detection
        _compute_fvg() — 3-candle imbalance pattern
            bullish FVG: candle[i-1].high < candle[i+1].low
            bearish FVG: candle[i-1].low  > candle[i+1].high
        _mark_filled_fvgs() — gap closed when price trades through entire range
        fvg_age_bars counted from formation candle

3.  FVG proximity merge
        _attach_nearest_fvg() — replaces liq proximity for sniper entry
        Attaches: fvg_high, fvg_low, fvg_side, fvg_filled, fvg_age_bars

4.  save_to_duckdb updated with all new columns

5.  price_position now computed in gold layer (was derived in strategy)
        Avoids redundant computation at decision time
"""

from __future__ import annotations

import logging
from datetime import timezone
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

if TYPE_CHECKING:
    from src.gold.duckdb_store import DuckDBStore

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
TIMEFRAMES: list[str]   = ["5min", "1h", "4h", "1d"]
RSI_PERIOD: int         = 14
ATR_PERIOD: int         = 14
BASE_SWING_WINDOW: int  = 5
MIN_SWING_ATR_RATIO     = 0.5
CONFLUENCE_TOLERANCE    = 0.002
SWEEP_TOLERANCE         = 0.0005
N_NEAREST_LEVELS        = 3
MAX_FVG_AGE_BARS        = 20     # discard FVGs older than this many 5-min candles

# Session windows (UTC hour, inclusive)
SESSIONS: dict[str, tuple[int, int]] = {
    "london":   (7,  16),
    "new_york": (12, 21),
    "asian":    (0,   7),
    "killzone": (7,   9),
}


class FeatureEngineer:
    """
    Computes professional-grade SMC features from a DataFrame of UnifiedTick rows.

    Usage
    -----
        fe = FeatureEngineer()
        enriched_df = fe.build_features(unified_ticks_df)
        fe.save_to_duckdb(enriched_df, store)
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def build_features(self, ticks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Full Scout & Sniper feature pipeline:
          1.  Resample ticks → OHLC (5-min primary + multi-TF for liquidity)
          2.  RSI(14) + ATR(14) on 5-min candles
          3.  Multi-TF liquidity levels (swing H/L + round numbers) — carried from v2
          4.  Swept level tracking
          5.  Confluence scoring
          6.  BOS / CHoCH structure break labels      ← NEW
          7.  Fair Value Gap detection + fill tracking ← NEW
          8.  Merge all features to tick granularity
          9.  Session labelling
          10. price_position tagging

        Returns tick-level DataFrame with columns:
            bar_open/high/low/close, rsi_14, atr_14,
            liq_level, liq_type, liq_side, liq_tf, liq_score,
            liq_confirmed, liq_swept,
            dist_to_nearest_high, dist_to_nearest_low,
            structure_direction, bos_detected, choch_detected,
            fvg_high, fvg_low, fvg_side, fvg_filled, fvg_age_bars,
            price_position, session
        """
        if ticks_df.empty:
            logger.warning("[FeatureEngineer] Empty DataFrame — skipping")
            return ticks_df

        ticks_df = ticks_df.copy()
        ticks_df["timestamp_utc"] = pd.to_datetime(ticks_df["timestamp_utc"], utc=True)
        ticks_df = ticks_df.sort_values("timestamp_utc")
        ticks_df["mid"] = (ticks_df["bid"] + ticks_df["ask"]) / 2.0

        # ── Step 1 & 2: 5-min candles + indicators ───────────────────────────
        candles_5m = self._resample_ohlc(ticks_df, "5min")
        candles_5m = self._compute_indicators(candles_5m)

        # ── Step 3: multi-TF liquidity levels (v2 logic, unchanged) ─────────
        all_liq: list[pd.DataFrame] = []
        for tf in TIMEFRAMES:
            candles_tf = self._resample_ohlc(ticks_df, tf)
            if len(candles_tf) < BASE_SWING_WINDOW * 2 + 1:
                logger.debug(f"[FeatureEngineer] Not enough candles for TF={tf}, skipping")
                continue
            candles_tf = self._compute_indicators(candles_tf)
            liq_tf = self._identify_liquidity_levels(candles_tf, tf)
            if not liq_tf.empty:
                all_liq.append(liq_tf)

        liq_df = pd.concat(all_liq, ignore_index=True) if all_liq else pd.DataFrame()

        # ── Step 4: swept levels ──────────────────────────────────────────────
        if not liq_df.empty:
            liq_df = self._mark_swept_levels(liq_df, candles_5m)

        # ── Step 5: confluence scoring ────────────────────────────────────────
        if not liq_df.empty:
            liq_df = self._score_confluence(liq_df)

        # ── Step 6: BOS / CHoCH on 5-min candles ─────────────────────────────
        candles_5m = self._compute_structure_breaks(candles_5m)

        # ── Step 7: FVG detection + fill tracking ─────────────────────────────
        fvg_df = self._compute_fvg(candles_5m)
        if not fvg_df.empty:
            fvg_df = self._mark_filled_fvgs(fvg_df, candles_5m)

        # ── Step 8: merge everything to tick resolution ───────────────────────
        enriched = self._merge_to_ticks(ticks_df, candles_5m, liq_df, fvg_df)

        # ── Step 9 & 10: session + price position ─────────────────────────────
        enriched = self._add_session_label(enriched)
        enriched = self._add_price_position(enriched)

        logger.info(
            f"[FeatureEngineer] Built features: {len(enriched)} ticks, "
            f"{len(candles_5m)} 5-min candles, "
            f"{len(liq_df)} liquidity levels, "
            f"{len(fvg_df)} FVGs"
        )
        return enriched

    def save_to_duckdb(self, df: pd.DataFrame, store: "DuckDBStore") -> None:
        """Persist enriched tick DataFrame to the Gold layer DuckDB store."""
        if df.empty:
            logger.warning("[FeatureEngineer] Nothing to save — DataFrame is empty")
            return
        cols = [
            # ── Identity ──────────────────────────────────────────────────────
            "timestamp_utc", "symbol", "bid", "ask", "mid", "volume", "volume_usd", "source",
            # ── 5-min candle ──────────────────────────────────────────────────
            "bar_open", "bar_high", "bar_low", "bar_close",
            # ── Indicators ───────────────────────────────────────────────────
            "rsi_14", "atr_14",
            # ── Liquidity (v2, unchanged) ─────────────────────────────────────
            "liq_level", "liq_type", "liq_side", "liq_tf",
            "liq_score", "liq_confirmed", "liq_swept",
            "dist_to_nearest_high", "dist_to_nearest_low",
            # ── Structure (v3 NEW) ────────────────────────────────────────────
            "structure_direction", "bos_detected", "choch_detected",
            # ── FVG (v3 NEW) ──────────────────────────────────────────────────
            "fvg_high", "fvg_low", "fvg_side", "fvg_timestamp", "fvg_filled", "fvg_age_bars",
            # ── Context ───────────────────────────────────────────────────────
            "price_position", "session",
        ]
        available = [c for c in cols if c in df.columns]
        store.upsert_features(df[available])

    # ── OHLC resampling ───────────────────────────────────────────────────────
    # Unchanged from v2

    def _resample_ohlc(self, ticks_df: pd.DataFrame, freq: str) -> pd.DataFrame:
        ohlc = (
            ticks_df.set_index("timestamp_utc")["mid"]
            .resample(freq)
            .ohlc()
            .dropna()
        )
        vol = (
            ticks_df.set_index("timestamp_utc")["volume"]
            .resample(freq)
            .sum()
        )
        candles = ohlc.join(vol, how="left")
        candles.index.name = "bar_time"
        candles.columns = ["bar_open", "bar_high", "bar_low", "bar_close", "bar_volume"]
        candles = candles.reset_index()
        candles["bar_time"] = pd.to_datetime(candles["bar_time"], utc=True)
        return candles

    # ── Indicators ────────────────────────────────────────────────────────────
    # Unchanged from v2

    def _compute_indicators(self, candles: pd.DataFrame) -> pd.DataFrame:
        candles = candles.copy()
        candles[f"rsi_{RSI_PERIOD}"] = RSIIndicator(
            close=candles["bar_close"], window=RSI_PERIOD
        ).rsi()
        candles[f"atr_{ATR_PERIOD}"] = AverageTrueRange(
            high=candles["bar_high"],
            low=candles["bar_low"],
            close=candles["bar_close"],
            window=ATR_PERIOD,
        ).average_true_range()
        return candles

    # ── BOS / CHoCH detection (NEW in v3) ─────────────────────────────────────

    def _compute_structure_breaks(self, candles: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Break of Structure (BOS) and Change of Character (CHoCH)
        on the 5-min candle series.

        Algorithm
        ─────────
        1.  Walk candles left-to-right maintaining a rolling list of
            confirmed swing highs and swing lows (same ATR-adaptive window
            as _identify_liquidity_levels).
        2.  When a candle's CLOSE breaks above the most recent confirmed
            swing high:
              - If current structure_direction is already "bullish" → BOS bullish
              - If current structure_direction is "bearish"         → CHoCH bullish
        3.  When a candle's CLOSE breaks below the most recent confirmed
            swing low:
              - If current structure_direction is already "bearish" → BOS bearish
              - If current structure_direction is "bullish"         → CHoCH bearish
        4.  structure_direction tracks the current market bias and is
            forward-filled so every candle carries a value.

        New columns added to candles:
            structure_direction  str   "bullish" | "bearish" | None
            bos_detected         bool
            choch_detected       bool
        """
        candles = candles.copy()
        n = len(candles)

        highs  = candles["bar_high"].values
        lows   = candles["bar_low"].values
        closes = candles["bar_close"].values

        atr_col = f"atr_{ATR_PERIOD}"
        avg_atr = float(candles[atr_col].median()) if atr_col in candles.columns else 0.0
        if avg_atr == 0 or np.isnan(avg_atr):
            avg_atr = (highs.max() - lows.min()) * 0.01

        structure_direction = [None] * n
        bos_detected        = [False] * n
        choch_detected      = [False] * n

        # Running confirmed swing references
        last_confirmed_swing_high: Optional[float] = None
        last_confirmed_swing_low:  Optional[float] = None
        current_direction: Optional[str] = None

        w = BASE_SWING_WINDOW

        for i in range(w, n):
            local_atr = candles[atr_col].iloc[i] if atr_col in candles.columns else avg_atr
            if np.isnan(local_atr) or local_atr == 0:
                local_atr = avg_atr

            # ── Update swing references (look-back only, no look-ahead needed) ─
            #    A confirmed swing high at bar j < i exists if:
            #    highs[j] == max of highs[j-w:j+w+1]  AND
            #    some close after j fell below lows[j]
            for j in range(max(0, i - w * 2), i - w + 1):
                lo_j = j - w
                hi_j = j + w + 1
                if lo_j < 0 or hi_j > n:
                    continue

                # Potential swing high
                if highs[j] >= highs[lo_j:hi_j].max() - 1e-8:
                    if self._confirm_swing_high(closes, lows, j, min(j + 21, i + 1)):
                        if (last_confirmed_swing_high is None
                                or highs[j] > last_confirmed_swing_high):
                            last_confirmed_swing_high = highs[j]

                # Potential swing low
                if lows[j] <= lows[lo_j:hi_j].min() + 1e-8:
                    if self._confirm_swing_low(closes, highs, j, min(j + 21, i + 1)):
                        if (last_confirmed_swing_low is None
                                or lows[j] < last_confirmed_swing_low):
                            last_confirmed_swing_low = lows[j]

            # ── Check for structural break ────────────────────────────────────
            if last_confirmed_swing_high is not None:
                if closes[i] > last_confirmed_swing_high:
                    if current_direction == "bullish":
                        bos_detected[i] = True        # continuation
                    else:
                        choch_detected[i] = True       # reversal
                    current_direction = "bullish"
                    last_confirmed_swing_high = None   # consumed — reset

            if last_confirmed_swing_low is not None:
                if closes[i] < last_confirmed_swing_low:
                    if current_direction == "bearish":
                        bos_detected[i] = True
                    else:
                        choch_detected[i] = True
                    current_direction = "bearish"
                    last_confirmed_swing_low = None

            structure_direction[i] = current_direction

        candles["structure_direction"] = structure_direction
        candles["bos_detected"]        = bos_detected
        candles["choch_detected"]      = choch_detected

        # Forward-fill direction so every candle carries the current bias
        candles["structure_direction"] = candles["structure_direction"].ffill()

        return candles

    # ── Fair Value Gap detection (NEW in v3) ──────────────────────────────────

    def _compute_fvg(self, candles: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Fair Value Gaps using the standard 3-candle pattern.

        Bullish FVG  (demand imbalance):
            candle[i-1].high  <  candle[i+1].low
            → gap range: [candle[i-1].high, candle[i+1].low]
            → candle[i] is a strong bullish impulse candle

        Bearish FVG  (supply imbalance):
            candle[i-1].low   >  candle[i+1].high
            → gap range: [candle[i+1].high, candle[i-1].low]
            → candle[i] is a strong bearish impulse candle

        Returns a DataFrame of all detected FVGs with columns:
            formed_at    (bar_time of the impulse candle[i])
            fvg_high     (upper boundary of the gap)
            fvg_low      (lower boundary of the gap)
            fvg_side     "bullish_fvg" | "bearish_fvg"
            fvg_filled   (False at detection — updated by _mark_filled_fvgs)
            fvg_age_bars (0 at detection — updated during merge)
        """
        n       = len(candles)
        highs   = candles["bar_high"].values
        lows    = candles["bar_low"].values
        times   = candles["bar_time"].values
        records = []

        for i in range(1, n - 1):
            prev_high = highs[i - 1]
            prev_low  = lows[i - 1]
            next_high = highs[i + 1]
            next_low  = lows[i + 1]

            # Bullish FVG: gap between prev candle top and next candle bottom
            if prev_high < next_low:
                records.append({
                    "formed_at":  times[i],
                    "fvg_high":   round(float(next_low),  5),
                    "fvg_low":    round(float(prev_high), 5),
                    "fvg_side":   "bullish_fvg",
                    "fvg_filled": False,
                })

            # Bearish FVG: gap between prev candle bottom and next candle top
            elif prev_low > next_high:
                records.append({
                    "formed_at":  times[i],
                    "fvg_high":   round(float(prev_low),  5),
                    "fvg_low":    round(float(next_high), 5),
                    "fvg_side":   "bearish_fvg",
                    "fvg_filled": False,
                })

        if not records:
            return pd.DataFrame(columns=[
                "formed_at", "fvg_high", "fvg_low",
                "fvg_side", "fvg_filled",
            ])

        fvg_df = pd.DataFrame(records)
        fvg_df["formed_at"] = pd.to_datetime(fvg_df["formed_at"], utc=True)
        return fvg_df

    def _mark_filled_fvgs(
        self, fvg_df: pd.DataFrame, candles: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Mark an FVG as filled when a subsequent candle's range fully
        overlaps the gap (price traded through the entire imbalance zone).

        Bullish FVG filled: a future candle's LOW <= fvg_low
            (price traded back down through the entire gap)
        Bearish FVG filled: a future candle's HIGH >= fvg_high
            (price traded back up through the entire gap)
        """
        fvg_df = fvg_df.copy()

        for idx, row in fvg_df.iterrows():
            formed_at = row["formed_at"]
            future    = candles[candles["bar_time"] > formed_at]

            if future.empty:
                continue

            if row["fvg_side"] == "bullish_fvg":
                filled = (future["bar_low"] <= row["fvg_low"]).any()
            else:
                filled = (future["bar_high"] >= row["fvg_high"]).any()

            fvg_df.at[idx, "fvg_filled"] = filled

        return fvg_df

    # ── Liquidity detection (unchanged from v2) ───────────────────────────────

    def _identify_liquidity_levels(
        self, candles: pd.DataFrame, timeframe: str
    ) -> pd.DataFrame:
        records: list[dict] = []
        highs  = candles["bar_high"].values
        lows   = candles["bar_low"].values
        closes = candles["bar_close"].values
        times  = candles["bar_time"].values

        atr_col = f"atr_{ATR_PERIOD}"
        avg_atr = candles[atr_col].median() if atr_col in candles.columns else 0.0
        if avg_atr == 0 or np.isnan(avg_atr):
            avg_atr = (highs.max() - lows.min()) * 0.01

        n = len(candles)

        for i in range(BASE_SWING_WINDOW, n - BASE_SWING_WINDOW):
            local_atr = candles[atr_col].iloc[i] if atr_col in candles.columns else avg_atr
            if np.isnan(local_atr) or local_atr == 0:
                local_atr = avg_atr

            w = self._dynamic_swing_window(local_atr, avg_atr)
            if i < w or i >= n - w:
                continue

            # Swing high
            window_highs = highs[i - w: i + w + 1]
            if highs[i] >= window_highs.max() - 1e-8:
                swing_size = highs[i] - lows[i]
                if swing_size >= local_atr * MIN_SWING_ATR_RATIO:
                    confirmed = self._confirm_swing_high(closes, lows, i, n)
                    records.append({
                        "bar_time":   times[i],
                        "liq_level":  round(float(highs[i]), 5),
                        "liq_type":   "swing_high",
                        "liq_side":   "buystops_above",
                        "timeframe":  timeframe,
                        "confirmed":  confirmed,
                        "swept":      False,
                        "swing_size": round(float(swing_size), 5),
                        "touch_count": 1,
                    })

            # Swing low
            window_lows = lows[i - w: i + w + 1]
            if lows[i] <= window_lows.min() + 1e-8:
                swing_size = highs[i] - lows[i]
                if swing_size >= local_atr * MIN_SWING_ATR_RATIO:
                    confirmed = self._confirm_swing_low(closes, highs, i, n)
                    records.append({
                        "bar_time":   times[i],
                        "liq_level":  round(float(lows[i]), 5),
                        "liq_type":   "swing_low",
                        "liq_side":   "sellstops_below",
                        "timeframe":  timeframe,
                        "confirmed":  confirmed,
                        "swept":      False,
                        "swing_size": round(float(swing_size), 5),
                        "touch_count": 1,
                    })

        # Round-number levels
        price_min = candles["bar_low"].min()
        price_max = candles["bar_high"].max()
        step      = self._adaptive_round_step(price_max)
        first_level = (int(price_min / step) + 1) * step

        for level in np.arange(first_level, price_max + step, step):
            level    = round(float(level), 2)
            bar_time = self._first_touch_time(candles, level, avg_atr)
            if bar_time is None:
                continue
            records.append({
                "bar_time":    bar_time,
                "liq_level":   level,
                "liq_type":    "round_number",
                "liq_side":    "buystops_above",
                "timeframe":   timeframe,
                "confirmed":   True,
                "swept":       False,
                "swing_size":  0.0,
                "touch_count": self._count_touches(candles, level, avg_atr),
            })

        if not records:
            return pd.DataFrame(columns=[
                "bar_time", "liq_level", "liq_type", "liq_side",
                "timeframe", "confirmed", "swept", "swing_size",
                "touch_count", "liq_score",
            ])

        liq_df = pd.DataFrame(records)
        liq_df["bar_time"] = pd.to_datetime(liq_df["bar_time"], utc=True)

        for idx, row in liq_df.iterrows():
            if row["liq_type"] != "round_number":
                liq_df.at[idx, "touch_count"] = self._count_touches(
                    candles, row["liq_level"], avg_atr
                )
        return liq_df

    # ── Structure break confirmation (unchanged from v2) ──────────────────────

    @staticmethod
    def _confirm_swing_high(
        closes: np.ndarray, lows: np.ndarray, i: int, n: int
    ) -> bool:
        swing_low  = lows[i]
        look_ahead = min(i + 21, n)
        return any(closes[j] < swing_low for j in range(i + 1, look_ahead))

    @staticmethod
    def _confirm_swing_low(
        closes: np.ndarray, highs: np.ndarray, i: int, n: int
    ) -> bool:
        swing_high = highs[i]
        look_ahead = min(i + 21, n)
        return any(closes[j] > swing_high for j in range(i + 1, look_ahead))

    # ── Swept level tracking (unchanged from v2) ──────────────────────────────

    def _mark_swept_levels(
        self, liq_df: pd.DataFrame, candles: pd.DataFrame
    ) -> pd.DataFrame:
        liq_df = liq_df.copy()
        liq_df["swept_at"] = pd.NaT
        closes = candles["bar_close"].values
        times  = candles["bar_time"].values

        for idx, row in liq_df.iterrows():
            level     = row["liq_level"]
            formed_at = row["bar_time"]
            side      = row["liq_side"]

            future_mask   = candles["bar_time"] > formed_at
            future_closes = closes[future_mask]
            future_times  = times[future_mask]

            if side == "buystops_above":
                hit = future_closes > level * (1 + SWEEP_TOLERANCE)
            else:
                hit = future_closes < level * (1 - SWEEP_TOLERANCE)

            if hit.any():
                liq_df.at[idx, "swept"]    = True
                liq_df.at[idx, "swept_at"] = future_times[np.argmax(hit)]
        return liq_df

    # ── Confluence scoring (unchanged from v2) ────────────────────────────────

    def _score_confluence(self, liq_df: pd.DataFrame) -> pd.DataFrame:
        liq_df = liq_df.copy()
        levels = liq_df["liq_level"].values
        scores = np.zeros(len(liq_df), dtype=float)

        for i, row in liq_df.iterrows():
            if row.get("swept", False):
                scores[i] = -99
                continue
            level = row["liq_level"]
            nearby   = np.abs(levels - level) / (level + 1e-10) <= CONFLUENCE_TOLERANCE
            tf_count = liq_df[nearby]["timeframe"].nunique()
            scores[i] += tf_count - 1
            if row.get("confirmed", False):
                scores[i] += 1
            scores[i] += max(0, int(row.get("touch_count", 1)) - 1)
            if row["timeframe"] == "1d":
                scores[i] += 3
            elif row["timeframe"] == "4h":
                scores[i] += 2
            elif row["timeframe"] == "1h":
                scores[i] += 1
            if pd.notna(row["bar_time"]):
                hour = pd.Timestamp(row["bar_time"]).hour
                kz_start, kz_end = SESSIONS["killzone"]
                if kz_start <= hour < kz_end:
                    scores[i] += 2

        liq_df["liq_score"] = scores
        return liq_df

    # ── Merge to tick resolution ──────────────────────────────────────────────

    def _merge_to_ticks(
        self,
        ticks_df: pd.DataFrame,
        candles:  pd.DataFrame,
        liq_df:   pd.DataFrame,
        fvg_df:   pd.DataFrame,
    ) -> pd.DataFrame:
        """
        1. Attach 5-min candle OHLC + indicators + structure columns → each tick
        2. Attach nearest unswept liquidity level above/below each tick
        3. Attach nearest active FVG to each tick
        """
        candles_sorted = candles.sort_values("bar_time")
        ticks_sorted   = ticks_df.sort_values("timestamp_utc")

        # ── Candle fields + structure fields ──────────────────────────────────
        candle_cols = [
            "bar_time", "bar_open", "bar_high", "bar_low", "bar_close",
            f"rsi_{RSI_PERIOD}", f"atr_{ATR_PERIOD}",
            "structure_direction", "bos_detected", "choch_detected",
        ]
        available_cols = [c for c in candle_cols if c in candles_sorted.columns]

        enriched = pd.merge_asof(
            ticks_sorted,
            candles_sorted[available_cols],
            left_on="timestamp_utc",
            right_on="bar_time",
            direction="backward",
        )
        enriched = enriched.rename(columns={
            f"rsi_{RSI_PERIOD}": "rsi_14",
            f"atr_{ATR_PERIOD}": "atr_14",
            "fvg_filled": "fvg_filled_candle",
        })

        # ── Liquidity attachment (unchanged from v2) ──────────────────────────
        if not liq_df.empty:
            active_liq = liq_df[
                (~liq_df["swept"]) & (liq_df["liq_score"] >= 0)
            ].copy()
            if not active_liq.empty:
                enriched = self._attach_nearest_levels(enriched, active_liq)
            else:
                enriched = self._add_empty_liq_columns(enriched)
        else:
            enriched = self._add_empty_liq_columns(enriched)

        # ── FVG attachment (NEW in v3) ────────────────────────────────────────
        if not fvg_df.empty:
            active_fvg = fvg_df[~fvg_df["fvg_filled"]].copy()
            if not active_fvg.empty:
                enriched = self._attach_nearest_fvg(enriched, active_fvg, candles_sorted)
            else:
                enriched = self._add_empty_fvg_columns(enriched)
        else:
            enriched = self._add_empty_fvg_columns(enriched)

        enriched.drop(columns=["mid", "bar_time"], errors="ignore", inplace=True)
        return enriched

    # ── Liquidity attachment (unchanged from v2) ──────────────────────────────

    def _attach_nearest_levels(
        self, enriched: pd.DataFrame, liq_df: pd.DataFrame
    ) -> pd.DataFrame:
        enriched  = enriched.copy()
        levels    = liq_df["liq_level"].values
        types     = liq_df["liq_type"].values
        sides     = liq_df["liq_side"].values
        tfs       = liq_df["timeframe"].values
        scores    = liq_df["liq_score"].values
        confirmed = liq_df["confirmed"].values
        swept     = liq_df["swept"].values

        nearest_level  = []
        nearest_type   = []
        nearest_side   = []
        nearest_tf     = []
        nearest_score  = []
        nearest_conf   = []
        nearest_swept_ = []
        dist_high      = []
        dist_low       = []

        for mid in enriched["mid"]:
            above_mask = levels > mid
            below_mask = levels < mid

            if above_mask.any():
                above_idx     = np.where(above_mask)[0]
                closest_above = above_idx[np.argmin(levels[above_idx] - mid)]
                dist_high.append(round(levels[closest_above] - mid, 5))
            else:
                closest_above = None
                dist_high.append(np.nan)

            if below_mask.any():
                below_idx     = np.where(below_mask)[0]
                closest_below = below_idx[np.argmin(mid - levels[below_idx])]
                dist_low.append(round(mid - levels[closest_below], 5))
            else:
                closest_below = None
                dist_low.append(np.nan)

            distances = np.abs(levels - mid)
            top_n_idx = np.argsort(distances)[:N_NEAREST_LEVELS]
            best_idx  = top_n_idx[np.argmax(scores[top_n_idx])]

            nearest_level.append(round(float(levels[best_idx]), 5))
            nearest_type.append(types[best_idx])
            nearest_side.append(sides[best_idx])
            nearest_tf.append(tfs[best_idx])
            nearest_score.append(float(scores[best_idx]))
            nearest_conf.append(bool(confirmed[best_idx]))
            nearest_swept_.append(bool(swept[best_idx]))

        enriched["liq_level"]            = nearest_level
        enriched["liq_type"]             = nearest_type
        enriched["liq_side"]             = nearest_side
        enriched["liq_tf"]               = nearest_tf
        enriched["liq_score"]            = nearest_score
        enriched["liq_confirmed"]        = nearest_conf
        enriched["liq_swept"]            = nearest_swept_
        enriched["dist_to_nearest_high"] = dist_high
        enriched["dist_to_nearest_low"]  = dist_low
        return enriched

    # ── FVG attachment (NEW in v3) ────────────────────────────────────────────

    def _attach_nearest_fvg(
        self,
        enriched:  pd.DataFrame,
        fvg_df:    pd.DataFrame,
        candles:   pd.DataFrame,
    ) -> pd.DataFrame:
        """
        For each tick, find the nearest ACTIVE (unfilled, within age limit)
        FVG whose side aligns with the current structure_direction.

        Alignment rule:
            bullish_fvg  →  only relevant when structure_direction == "bullish"
            bearish_fvg  →  only relevant when structure_direction == "bearish"

        Also computes fvg_age_bars = number of 5-min candles since the FVG formed.
        """
        enriched = enriched.copy()

        fvg_highs     = fvg_df["fvg_high"].values
        fvg_lows      = fvg_df["fvg_low"].values
        fvg_sides     = fvg_df["fvg_side"].values
        fvg_formed_at = pd.to_datetime(fvg_df["formed_at"].values, utc=True)

        # Map each bar_time to its candle index for age calculation
        bar_times = pd.to_datetime(candles["bar_time"].values, utc=True)

        out_fvg_high  = []
        out_fvg_low   = []
        out_fvg_side  = []
        out_fvg_filled = []
        out_fvg_age   = []

        for _, row in enriched.iterrows():
            mid       = row["mid"]
            tick_time = row["timestamp_utc"]
            direction = row.get("structure_direction")

            # Filter by alignment + age
            valid_mask = np.ones(len(fvg_df), dtype=bool)
            for k in range(len(fvg_df)):
                # Side must align with structure direction
                if direction == "bullish" and fvg_sides[k] != "bullish_fvg":
                    valid_mask[k] = False
                    continue
                if direction == "bearish" and fvg_sides[k] != "bearish_fvg":
                    valid_mask[k] = False
                    continue
                # FVG must have formed before this tick
                if fvg_formed_at[k] >= tick_time:
                    valid_mask[k] = False
                    continue
                # Age check: count 5-min candles since formation
                formed_idx = np.searchsorted(bar_times, fvg_formed_at[k], side="right")
                tick_idx   = np.searchsorted(bar_times, tick_time, side="right")
                age_bars   = tick_idx - formed_idx
                if age_bars > MAX_FVG_AGE_BARS:
                    valid_mask[k] = False

            valid_indices = np.where(valid_mask)[0]

            if len(valid_indices) == 0:
                out_fvg_high.append(np.nan)
                out_fvg_low.append(np.nan)
                out_fvg_side.append(None)
                out_fvg_filled.append(None)
                out_fvg_age.append(None)
                continue

            # Nearest FVG by midpoint distance
            fvg_mids  = (fvg_highs[valid_indices] + fvg_lows[valid_indices]) / 2.0
            distances = np.abs(fvg_mids - mid)
            best      = valid_indices[np.argmin(distances)]

            formed_idx = np.searchsorted(bar_times, fvg_formed_at[best], side="right")
            tick_idx   = np.searchsorted(bar_times, tick_time, side="right")
            age_bars   = int(tick_idx - formed_idx)

            out_fvg_high.append(round(float(fvg_highs[best]), 5))
            out_fvg_low.append(round(float(fvg_lows[best]), 5))
            out_fvg_side.append(fvg_sides[best])
            out_fvg_filled.append(bool(fvg_df.iloc[best]["fvg_filled"]))
            out_fvg_age.append(age_bars)

        enriched["fvg_high"]     = out_fvg_high
        enriched["fvg_low"]      = out_fvg_low
        enriched["fvg_side"]     = out_fvg_side
        enriched["fvg_filled"]   = out_fvg_filled
        enriched["fvg_age_bars"] = out_fvg_age
        return enriched

    @staticmethod
    def _add_empty_liq_columns(df: pd.DataFrame) -> pd.DataFrame:
        for col in [
            "liq_level", "liq_type", "liq_side", "liq_tf",
            "liq_score", "liq_confirmed", "liq_swept",
            "dist_to_nearest_high", "dist_to_nearest_low",
        ]:
            df[col] = None
        return df

    @staticmethod
    def _add_empty_fvg_columns(df: pd.DataFrame) -> pd.DataFrame:
        for col in ["fvg_high", "fvg_low", "fvg_side", "fvg_filled", "fvg_age_bars"]:
            df[col] = None
        return df

    # ── Session labelling (unchanged from v2) ─────────────────────────────────

    @staticmethod
    def _add_session_label(df: pd.DataFrame) -> pd.DataFrame:
        def label(ts: pd.Timestamp) -> str:
            h = ts.hour
            if SESSIONS["killzone"][0] <= h < SESSIONS["killzone"][1]:
                return "killzone"
            if SESSIONS["london"][0] <= h < SESSIONS["london"][1]:
                return "london"
            if SESSIONS["new_york"][0] <= h < SESSIONS["new_york"][1]:
                return "new_york"
            return "asian"
        df = df.copy()
        df["session"] = df["timestamp_utc"].apply(label)
        return df

    # ── Price position (moved from strategy into gold layer in v3) ────────────

    @staticmethod
    def _add_price_position(df: pd.DataFrame) -> pd.DataFrame:
        """
        Tag each tick with its premium/discount zone using dist columns.
        Moved from strategy.py → gold layer to avoid redundant computation
        at decision time.

        dist_low / (dist_high + dist_low):
            < 0.25  → discount_extreme
            < 0.50  → discount
            < 0.75  → premium
            >= 0.75 → premium_extreme
        """
        def derive(row) -> Optional[str]:
            dh = row.get("dist_to_nearest_high")
            dl = row.get("dist_to_nearest_low")
            if pd.isna(dh) or pd.isna(dl):
                return None
            total = dh + dl
            if total == 0:
                return None
            ratio = dl / total
            if ratio < 0.25:
                return "discount_extreme"
            if ratio < 0.50:
                return "discount"
            if ratio < 0.75:
                return "premium"
            return "premium_extreme"

        df = df.copy()
        df["price_position"] = df.apply(derive, axis=1)
        return df

    # ── Helper utilities (unchanged from v2) ──────────────────────────────────

    @staticmethod
    def _dynamic_swing_window(atr: float, avg_atr: float) -> int:
        if avg_atr == 0:
            return BASE_SWING_WINDOW
        ratio = atr / avg_atr
        if ratio > 1.5:
            return 8
        if ratio < 0.5:
            return 3
        return BASE_SWING_WINDOW

    @staticmethod
    def _adaptive_round_step(price: float) -> float:
        if price < 10:
            return 0.10
        if price < 100:
            return 1.0
        if price < 500:
            return 5.0
        if price < 2000:
            return 10.0
        return 50.0

    @staticmethod
    def _first_touch_time(
        candles: pd.DataFrame, level: float, atr: float
    ) -> pd.Timestamp | None:
        tolerance = atr * 0.5
        mask = (
            (candles["bar_low"]  <= level + tolerance) &
            (candles["bar_high"] >= level - tolerance)
        )
        hits = candles[mask]
        return hits["bar_time"].iloc[0] if not hits.empty else None

    @staticmethod
    def _count_touches(
        candles: pd.DataFrame, level: float, atr: float
    ) -> int:
        tolerance = atr * 0.5
        mask = (
            (candles["bar_high"] >= level - tolerance) &
            (candles["bar_low"]  <= level + tolerance)
        )
        return int(mask.sum())