"""
Gold Layer — FeatureEngineer  (v2 — Professional Liquidity Rebuild)

What changed from v1
────────────────────
1.  Multi-timeframe swing detection  (5min, 1H, 4H, 1D)
2.  Structure-break confirmation before a swing is labelled valid
3.  Stop-side awareness  (buystops_above / sellstops_below)
4.  Swept / consumed level tracking  — invalidated levels are excluded
5.  Session-aware weighting  (London / NY killzone levels score higher)
6.  Level confluence scoring  (same price across TFs = stronger level)
7.  Distance matrix per tick  (nearest N levels above AND below)
8.  ATR-relative minimum swing size filter  (noise rejection)
9.  ATR-adaptive swing window  (widens in high-vol, narrows in low-vol)
10. Adaptive round-number step  (scales with price magnitude)
11. Float-safe swing comparison  (tolerance 1e-8, no missed peaks)
12. Round numbers get realistic timestamps  (first candle near the level)
"""

from __future__ import annotations

import logging
from datetime import timezone
from typing import TYPE_CHECKING, Literal

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
BASE_SWING_WINDOW: int  = 5      # bars each side at default volatility
MIN_SWING_ATR_RATIO     = 0.5    # swing must be ≥ 50 % of ATR to be meaningful
CONFLUENCE_TOLERANCE    = 0.002  # 0.2 % — levels within this band are confluent
SWEEP_TOLERANCE         = 0.0005 # 0.05 % — price must close beyond level to sweep it
N_NEAREST_LEVELS        = 3      # levels above + below attached to each tick

# Session windows (UTC hour, inclusive)
SESSIONS: dict[str, tuple[int, int]] = {
    "london":    (7,  16),
    "new_york":  (12, 21),
    "asian":     (0,   7),
    "killzone":  (7,   9),   # London open killzone (highest institutional activity)
}


class FeatureEngineer:
    """
    Computes professional-grade features from a DataFrame of UnifiedTick rows.

    Usage
    -----
        fe = FeatureEngineer()
        enriched_df = fe.build_features(unified_ticks_df)
        fe.save_to_duckdb(enriched_df, store)
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def build_features(self, ticks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Full feature pipeline:
          1.  Resample ticks to multiple OHLC timeframes
          2.  Compute RSI(14) + ATR(14) on 5-min candles
          3.  Identify & confirm liquidity levels (multi-TF)
          4.  Track swept / consumed levels
          5.  Score confluence across timeframes
          6.  Merge distance matrix back to tick granularity

        Returns tick-level DataFrame with columns:
            bar_open/high/low/close, rsi_14, atr_14,
            liq_level, liq_type, liq_side, liq_tf, liq_score,
            liq_confirmed, liq_swept,
            dist_to_nearest_high, dist_to_nearest_low,
            session
        """
        if ticks_df.empty:
            logger.warning("[FeatureEngineer] Empty DataFrame — skipping")
            return ticks_df

        ticks_df = ticks_df.copy()
        ticks_df["timestamp_utc"] = pd.to_datetime(ticks_df["timestamp_utc"], utc=True)
        ticks_df = ticks_df.sort_values("timestamp_utc")
        ticks_df["mid"] = (ticks_df["bid"] + ticks_df["ask"]) / 2.0

        # ── Step 1 & 2: candles + indicators (primary 5-min frame) ───────────
        candles_5m = self._resample_ohlc(ticks_df, "5min")
        candles_5m = self._compute_indicators(candles_5m)

        # ── Step 3: multi-TF candles for swing detection ─────────────────────
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

        # ── Step 4: mark swept levels ─────────────────────────────────────────
        if not liq_df.empty:
            liq_df = self._mark_swept_levels(liq_df, candles_5m)

        # ── Step 5: score confluence ──────────────────────────────────────────
        if not liq_df.empty:
            liq_df = self._score_confluence(liq_df)

        # ── Step 6: merge to ticks ────────────────────────────────────────────
        enriched = self._merge_to_ticks(ticks_df, candles_5m, liq_df)
        enriched = self._add_session_label(enriched)

        logger.info(
            f"[FeatureEngineer] Built features: {len(enriched)} ticks, "
            f"{len(candles_5m)} 5-min candles, "
            f"{len(liq_df)} liquidity levels across {len(TIMEFRAMES)} TFs"
        )
        return enriched

    def save_to_duckdb(self, df: pd.DataFrame, store: "DuckDBStore") -> None:
        """Persist enriched tick DataFrame to the Gold layer DuckDB store."""
        if df.empty:
            logger.warning("[FeatureEngineer] Nothing to save — DataFrame is empty")
            return
        cols = [
            "timestamp_utc", "symbol", "bid", "ask", "volume", "volume_usd", "source",
            "bar_open", "bar_high", "bar_low", "bar_close",
            "rsi_14", "atr_14",
            "liq_level", "liq_type", "liq_side", "liq_tf",
            "liq_score", "liq_confirmed", "liq_swept",
            "dist_to_nearest_high", "dist_to_nearest_low",
            "session",
        ]
        available = [c for c in cols if c in df.columns]
        store.upsert_features(df[available])

    # ── OHLC resampling ───────────────────────────────────────────────────────

    def _resample_ohlc(self, ticks_df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Resample mid-price ticks to OHLC candles at the given frequency."""
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

    def _compute_indicators(self, candles: pd.DataFrame) -> pd.DataFrame:
        """RSI(14) on close, ATR(14) on high/low/close."""
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

    # ── Liquidity detection ───────────────────────────────────────────────────

    def _identify_liquidity_levels(
        self, candles: pd.DataFrame, timeframe: str
    ) -> pd.DataFrame:
        """
        Detect swing highs/lows and round-number levels on a given TF candle set.

        Improvements over v1
        ────────────────────
        - ATR-adaptive swing window (wider in high-vol, tighter in low-vol)
        - ATR-relative minimum swing size (ignores noise micro-swings)
        - Float-safe comparison (1e-8 tolerance)
        - Structure-break confirmation (swing is only valid if a subsequent
          candle closes on the opposite side)
        - Stop-side correctly assigned per level type
        - Round-number step adapts to price magnitude
        - Round-number bar_time = first candle that touched the level
        """
        records: list[dict] = []
        highs  = candles["bar_high"].values
        lows   = candles["bar_low"].values
        closes = candles["bar_close"].values
        times  = candles["bar_time"].values

        atr_col = f"atr_{ATR_PERIOD}"
        avg_atr = candles[atr_col].median() if atr_col in candles.columns else 0.0
        if avg_atr == 0 or np.isnan(avg_atr):
            avg_atr = (highs.max() - lows.min()) * 0.01   # fallback: 1% of range

        n = len(candles)

        for i in range(BASE_SWING_WINDOW, n - BASE_SWING_WINDOW):
            local_atr = candles[atr_col].iloc[i] if atr_col in candles.columns else avg_atr
            if np.isnan(local_atr) or local_atr == 0:
                local_atr = avg_atr

            w = self._dynamic_swing_window(local_atr, avg_atr)
            if i < w or i >= n - w:
                continue

            # ── Swing high ───────────────────────────────────────────────────
            window_highs = highs[i - w: i + w + 1]
            local_max    = window_highs.max()

            if highs[i] >= local_max - 1e-8:
                swing_size = highs[i] - lows[i]

                # Noise filter: swing must be meaningful relative to ATR
                if swing_size >= local_atr * MIN_SWING_ATR_RATIO:

                    # Structure-break confirmation: a later candle must close
                    # BELOW the swing low of the originating bar to confirm
                    confirmed = self._confirm_swing_high(closes, lows, i, n)

                    records.append({
                        "bar_time":    times[i],
                        "liq_level":   round(float(highs[i]), 5),
                        "liq_type":    "swing_high",
                        "liq_side":    "buystops_above",
                        "timeframe":   timeframe,
                        "confirmed":   confirmed,
                        "swept":       False,
                        "swing_size":  round(float(swing_size), 5),
                        "touch_count": 1,
                    })

            # ── Swing low ────────────────────────────────────────────────────
            window_lows = lows[i - w: i + w + 1]
            local_min   = window_lows.min()

            if lows[i] <= local_min + 1e-8:
                swing_size = highs[i] - lows[i]

                if swing_size >= local_atr * MIN_SWING_ATR_RATIO:
                    confirmed = self._confirm_swing_low(closes, highs, i, n)

                    records.append({
                        "bar_time":    times[i],
                        "liq_level":   round(float(lows[i]), 5),
                        "liq_type":    "swing_low",
                        "liq_side":    "sellstops_below",
                        "timeframe":   timeframe,
                        "confirmed":   confirmed,
                        "swept":       False,
                        "swing_size":  round(float(swing_size), 5),
                        "touch_count": 1,
                    })

        # ── Round-number levels ───────────────────────────────────────────────
        price_min = candles["bar_low"].min()
        price_max = candles["bar_high"].max()
        step      = self._adaptive_round_step(price_max)

        first_level = (int(price_min / step) + 1) * step
        for level in np.arange(first_level, price_max + step, step):
            level = round(float(level), 2)
            # Assign bar_time = first candle that came within 0.5 ATR
            bar_time = self._first_touch_time(candles, level, avg_atr)
            if bar_time is None:
                continue

            records.append({
                "bar_time":    bar_time,
                "liq_level":   level,
                "liq_type":    "round_number",
                "liq_side":    "buystops_above",   # conservative default
                "timeframe":   timeframe,
                "confirmed":   True,               # round numbers are always valid
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

        # ── Touch count on swing levels ───────────────────────────────────────
        for idx, row in liq_df.iterrows():
            if row["liq_type"] != "round_number":
                liq_df.at[idx, "touch_count"] = self._count_touches(
                    candles, row["liq_level"], avg_atr
                )

        return liq_df

    # ── Structure break confirmation ──────────────────────────────────────────

    @staticmethod
    def _confirm_swing_high(
        closes: np.ndarray, lows: np.ndarray, i: int, n: int
    ) -> bool:
        """
        A swing high is confirmed when a subsequent candle closes BELOW
        the low of the swing candle itself (market structure break downward).
        Look ahead up to 20 bars.
        """
        swing_low = lows[i]
        look_ahead = min(i + 21, n)
        return any(closes[j] < swing_low for j in range(i + 1, look_ahead))

    @staticmethod
    def _confirm_swing_low(
        closes: np.ndarray, highs: np.ndarray, i: int, n: int
    ) -> bool:
        """
        A swing low is confirmed when a subsequent candle closes ABOVE
        the high of the swing candle (market structure break upward).
        """
        swing_high = highs[i]
        look_ahead = min(i + 21, n)
        return any(closes[j] > swing_high for j in range(i + 1, look_ahead))

    # ── Swept level tracking ──────────────────────────────────────────────────

    def _mark_swept_levels(
        self, liq_df: pd.DataFrame, candles: pd.DataFrame
    ) -> pd.DataFrame:
        """
        For each liquidity level, check whether a later 5-min candle closed
        beyond the level (i.e. liquidity was consumed / stop run completed).

        buystops_above  → swept when a candle CLOSES above liq_level
        sellstops_below → swept when a candle CLOSES below liq_level
        """
        liq_df = liq_df.copy()
        liq_df["swept_at"] = pd.NaT

        closes = candles["bar_close"].values
        times  = candles["bar_time"].values

        for idx, row in liq_df.iterrows():
            level      = row["liq_level"]
            formed_at  = row["bar_time"]
            side       = row["liq_side"]

            # Only look at candles after the level was formed
            future_mask = candles["bar_time"] > formed_at
            future_closes = closes[future_mask]
            future_times  = times[future_mask]

            if side == "buystops_above":
                hit = future_closes > level * (1 + SWEEP_TOLERANCE)
            else:   # sellstops_below
                hit = future_closes < level * (1 - SWEEP_TOLERANCE)

            if hit.any():
                liq_df.at[idx, "swept"]    = True
                liq_df.at[idx, "swept_at"] = future_times[np.argmax(hit)]

        return liq_df

    # ── Confluence scoring ────────────────────────────────────────────────────

    def _score_confluence(self, liq_df: pd.DataFrame) -> pd.DataFrame:
        """
        Score each level by how many other levels (across all TFs) are within
        CONFLUENCE_TOLERANCE of the same price.

        Score components:
          +1  per confirming level from another timeframe
          +1  if confirmed (structure break observed)
          +1  per touch_count beyond the first
          +2  if formed during London/NY killzone
          +3  if level is on the daily timeframe
          -99 if swept (invalidated — excluded from bot signals)
        """
        liq_df = liq_df.copy()
        levels = liq_df["liq_level"].values
        scores = np.zeros(len(liq_df), dtype=float)

        for i, row in liq_df.iterrows():
            if row.get("swept", False):
                scores[i] = -99
                continue

            level = row["liq_level"]

            # Cross-TF confluence
            nearby = np.abs(levels - level) / (level + 1e-10) <= CONFLUENCE_TOLERANCE
            tf_count = liq_df[nearby]["timeframe"].nunique()
            scores[i] += tf_count - 1   # +1 per additional TF

            # Confirmation bonus
            if row.get("confirmed", False):
                scores[i] += 1

            # Touch count bonus
            scores[i] += max(0, int(row.get("touch_count", 1)) - 1)

            # Timeframe weight
            if row["timeframe"] == "1d":
                scores[i] += 3
            elif row["timeframe"] == "4h":
                scores[i] += 2
            elif row["timeframe"] == "1h":
                scores[i] += 1

            # Killzone formation bonus
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
        candles: pd.DataFrame,
        liq_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        1. Attach 5-min candle OHLC + indicators to each tick (merge_asof backward).
        2. Attach nearest unswept liquidity level ABOVE and BELOW each tick
           (by price proximity, not by time).
        3. Compute distance to nearest high/low level.
        """
        # ── candle fields ─────────────────────────────────────────────────────
        candles_sorted = candles.sort_values("bar_time")
        ticks_sorted   = ticks_df.sort_values("timestamp_utc")

        enriched = pd.merge_asof(
            ticks_sorted,
            candles_sorted[[
                "bar_time", "bar_open", "bar_high", "bar_low", "bar_close",
                f"rsi_{RSI_PERIOD}", f"atr_{ATR_PERIOD}",
            ]],
            left_on="timestamp_utc",
            right_on="bar_time",
            direction="backward",
        )
        enriched = enriched.rename(columns={
            f"rsi_{RSI_PERIOD}": "rsi_14",
            f"atr_{ATR_PERIOD}": "atr_14",
        })

        # ── price-proximity liquidity attachment ──────────────────────────────
        if not liq_df.empty:
            # Only work with unswept, confirmed levels for signal generation
            active_liq = liq_df[
                (~liq_df["swept"]) & (liq_df["liq_score"] >= 0)
            ].copy()

            if not active_liq.empty:
                enriched = self._attach_nearest_levels(enriched, active_liq)
            else:
                enriched = self._add_empty_liq_columns(enriched)
        else:
            enriched = self._add_empty_liq_columns(enriched)

        enriched.drop(
            columns=["mid", "bar_time"], errors="ignore", inplace=True
        )
        return enriched

    def _attach_nearest_levels(
        self, enriched: pd.DataFrame, liq_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        For each tick, find:
          - The nearest unswept level ABOVE mid price (buy-stop zone)
          - The nearest unswept level BELOW mid price (sell-stop zone)
          - Distance to each in price units
          - The strongest level (highest liq_score) within N_NEAREST_LEVELS
        """
        enriched = enriched.copy()
        levels    = liq_df["liq_level"].values
        types     = liq_df["liq_type"].values
        sides     = liq_df["liq_side"].values
        tfs       = liq_df["timeframe"].values
        scores    = liq_df["liq_score"].values
        confirmed = liq_df["confirmed"].values
        swept     = liq_df["swept"].values

        nearest_level   = []
        nearest_type    = []
        nearest_side    = []
        nearest_tf      = []
        nearest_score   = []
        nearest_conf    = []
        nearest_swept_  = []
        dist_high       = []
        dist_low        = []

        for mid in enriched["mid"]:
            above_mask = levels > mid
            below_mask = levels < mid

            # Nearest above
            if above_mask.any():
                above_idx   = np.where(above_mask)[0]
                closest_above = above_idx[np.argmin(levels[above_idx] - mid)]
                dist_high.append(round(levels[closest_above] - mid, 5))
            else:
                closest_above = None
                dist_high.append(np.nan)

            # Nearest below
            if below_mask.any():
                below_idx   = np.where(below_mask)[0]
                closest_below = below_idx[np.argmin(mid - levels[below_idx])]
                dist_low.append(round(mid - levels[closest_below], 5))
            else:
                closest_below = None
                dist_low.append(np.nan)

            # Primary level: highest-scoring among N nearest (above or below)
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

        enriched["liq_level"]           = nearest_level
        enriched["liq_type"]            = nearest_type
        enriched["liq_side"]            = nearest_side
        enriched["liq_tf"]              = nearest_tf
        enriched["liq_score"]           = nearest_score
        enriched["liq_confirmed"]       = nearest_conf
        enriched["liq_swept"]           = nearest_swept_
        enriched["dist_to_nearest_high"] = dist_high
        enriched["dist_to_nearest_low"]  = dist_low
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

    # ── Session labelling ─────────────────────────────────────────────────────

    @staticmethod
    def _add_session_label(df: pd.DataFrame) -> pd.DataFrame:
        """Tag each tick with its trading session (killzone > london > new_york > asian)."""
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

    # ── Helper utilities ──────────────────────────────────────────────────────

    @staticmethod
    def _dynamic_swing_window(atr: float, avg_atr: float) -> int:
        """
        Widen the confirmation window in high-volatility regimes,
        tighten it in quiet markets.
        """
        if avg_atr == 0:
            return BASE_SWING_WINDOW
        ratio = atr / avg_atr
        if ratio > 1.5:
            return 8    # high vol — wider confirmation needed
        if ratio < 0.5:
            return 3    # low vol — tighter is fine
        return BASE_SWING_WINDOW

    @staticmethod
    def _adaptive_round_step(price: float) -> float:
        """
        Return a round-number step that makes sense for the current price level.
        Prevents the v1 bug where ROUND_NUMBER_STEP=10 produced zero levels
        for scaled historical data (e.g. price ~4.20).
        """
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
        """
        Return the bar_time of the first candle whose range includes
        the round-number level within 0.5 × ATR.
        Fixes v1 bug where all round numbers received iloc[0] timestamp.
        """
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
        """
        Count candles that came within 0.5 × ATR of the level.
        Higher touch_count → stronger / more contested level.
        """
        tolerance = atr * 0.5
        mask = (
            (candles["bar_high"] >= level - tolerance) &
            (candles["bar_low"]  <= level + tolerance)
        )
        return int(mask.sum())