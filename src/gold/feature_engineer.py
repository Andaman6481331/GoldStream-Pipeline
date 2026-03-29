"""
Gold Layer — FeatureEngineer (Scout & Sniper SMC Strategy)
────────────────────
Refactored to align with "Scout & Sniper" SMC-based strategy (1m / 15m / 4H).

1.  Structural Node State Machine (15m)
        _compute_smc_structure_nodes() — Tracks HH, LL, StrongHigh, StrongLow, CHoCH, and BOS.
2.  Fair Value Gap (FVG) detection (1m)
        _compute_fvg_smc() — 3-candle imbalance with Displacement (Gate 1) and Size (Gate 2) filters.
3.  Tiered Liquidity Levels
        Tier 1: Prev Day High/Low
        Tier 2: Session High/Low (Asian, London, NY)
        Tier 3: Structural Swings + EQH/EQL Clusters
4.  4H Market Bias
        _compute_4h_market_bias() — Higher timeframe trend filter.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import timezone
from typing import TYPE_CHECKING, Literal, Optional, List, Dict, Union, Any, cast, Tuple

import numpy as np
import pandas as pd
from ta.volatility import AverageTrueRange

if TYPE_CHECKING:
    from src.gold.duckdb_store import DuckDBStore

logger = logging.getLogger(__name__)

# ── Configuration (SMC Strategy Parameters) ──────────────────────────────────
# ATR Periods
ATR_PERIOD_1M:  int = 20   # atr_20_1m  — FVG size filter, displacement check
ATR_PERIOD_15M: int = 15   # atr_15_15m — R_dynamic, sweep tolerance, P1/P2

# Williams Fractal parameters
FRACTAL_L: int = 5         # lookback window — fixed (4H and 15m)
FRACTAL_R_MAX: int = 5     # maximum forward confirmation window (R_dynamic clamp max)
FRACTAL_R_MIN: int = 2     # minimum forward confirmation window (R_dynamic clamp min)

# 1m Fractal (L=R=3 fixed for trailing SL)
FRACTAL_1M_L: int = 3
FRACTAL_1M_R: int = 3

# R_dynamic calibration — k = R_DYNAMIC_K_FACTOR × avg_atr_15m
R_DYNAMIC_K_FACTOR: float = 3.0

# SMC Gates & Scoring
DISPLACEMENT_FACTOR: float = 0.5   # Gate 1: impulse body vs ATR
FVG_ATR_MULTIPLIER:  float = 0.15  # Gate 2: min FVG size (suggested 0.10-0.20)
ABSOLUTE_MIN_PIPS:   float = 3.0   # Gate 2: absolute floor for FVG
SWEEP_ATR_FACTOR:    float = 0.15  # Gate 4: sweep tolerance
SWEEP_MIN_PIPS:      float = 2.0   # Gate 4: absolute floor for sweep

# Tier 3 Liquidity (EQH/EQL)
MIN_CLUSTER_SIZE: int = 2
EQUAL_HL_ATR_FACTOR: float = 0.25
EQUAL_HL_LOOKBACK: int = 80
CONSUMPTION_LOOKBACK: int = 5

# FVG configuration
MAX_FVG_AGE_BARS: int = 20     # discard FVGs older than this many candles

# Session boundaries (UTC hour)
SESSION_BOUNDARIES: dict[str, int] = {
    "asian":  0,
    "london": 8,
    "ny":     13,
}

# Session windows for labelling
SESSIONS: dict[str, tuple[int, int]] = {
    "london":   (7,  16),
    "new_york": (12, 21),
    "asian":    (0,   7),
    "killzone": (7,   9),
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class SwingPoint:
    """A confirmed Williams Fractal swing point on the 15m chart."""
    price:    float
    bar_time: pd.Timestamp
    kind:     Literal["high", "low"]
    bar_idx:  int


@dataclass
class LiquidityLevel:
    """Tiered liquidity level as per strategy Phase 3 Gate 4."""
    price:    float
    tier:     int        # 1 | 2 | 3
    kind:     Literal["high", "low"]
    label:    str        # e.g. "Prev Day High", "EQH Cluster"
    strength: int = 1    # swing point count (EQH/EQL only)


@dataclass
class SessionSnapshot:
    """Price levels captured at a session boundary."""
    snapshot_time:        pd.Timestamp
    session_name:         str
    prev_day_high:        float
    prev_day_low:         float
    current_session_high: float
    current_session_low:  float
    prev_session_high:    float
    prev_session_low:     float


# ── Main class ────────────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Computes professional-grade SMC features from a DataFrame of UnifiedTick rows.
    """

    def __init__(self) -> None:
        self.swing_highs_15m: list[SwingPoint] = []
        self.swing_lows_15m:  list[SwingPoint] = []
        self.atr_15m_avg:     float = 0.0
        self.atr_1m_avg:      float = 0.0
        
        # Multi-TF Candle Store
        self.candles_1m:  pd.DataFrame = pd.DataFrame()
        self.candles_15m: pd.DataFrame = pd.DataFrame()
        self.candles_4h:  pd.DataFrame = pd.DataFrame()

    # ── Public API ────────────────────────────────────────────────────────────

    def build_features(self, ticks_df: pd.DataFrame) -> pd.DataFrame:
        """Full Scout & Sniper SMC feature pipeline."""
        self.candles_1m  = pd.DataFrame()  # reset/safe defaults
        self.candles_15m = pd.DataFrame()
        self.candles_4h  = pd.DataFrame()

        if ticks_df.empty:
            logger.warning("[FeatureEngineer] Empty DataFrame — skipping")
            return ticks_df

        ticks_df = ticks_df.copy()
        ticks_df["timestamp_utc"] = pd.to_datetime(ticks_df["timestamp_utc"], utc=True)
        ticks_df = ticks_df.sort_values("timestamp_utc")
        ticks_df["mid"] = (ticks_df["bid"] + ticks_df["ask"]) / 2.0
        symbol = ticks_df["symbol"].iloc[0]
        source = ticks_df["source"].iloc[0]

        # Step 1: Resample SMC timeframes
        self.candles_1m  = self._resample_ohlc(ticks_df, "1min")
        self.candles_15m = self._resample_ohlc(ticks_df, "15min")
        self.candles_4h  = self._resample_ohlc(ticks_df, "4h")

        if not self.candles_1m.empty:
            self.candles_1m["symbol"] = symbol
            self.candles_1m["source"] = source
        if not self.candles_15m.empty:
            self.candles_15m["symbol"] = symbol
            self.candles_15m["source"] = source
        if not self.candles_4h.empty:
            self.candles_4h["symbol"] = symbol
            self.candles_4h["source"] = source

        # Step 2: SMC Indicators (ATR)
        self.candles_1m  = self._compute_smc_atr(self.candles_1m,  ATR_PERIOD_1M,  "atr_20_1m")
        self.candles_15m = self._compute_smc_atr(self.candles_15m, ATR_PERIOD_15M, "atr_15_15m")

        self.atr_15m_avg = float(self.candles_15m["atr_15_15m"].median() or 0.0)
        self.atr_1m_avg  = float(self.candles_1m["atr_20_1m"].median() or 0.0)

        # Step 3: Session level tracker (Tier 1 & 2)
        session_levels_df = self._build_session_levels(ticks_df, self.candles_4h)

        # Step 4: 15m confirmed swing history
        self.swing_highs_15m, self.swing_lows_15m = self._build_swing_history_15m(self.candles_15m)
        swing_count_df = self._build_swing_count_series(
            ticks_df, self.candles_15m, self.swing_highs_15m, self.swing_lows_15m
        )

        # Step 5: SMC Structural Node State Machine (15m)
        structure_df = self._compute_smc_structure_nodes(
            self.candles_15m, self.swing_highs_15m, self.swing_lows_15m
        )
        if not structure_df.empty:
             # Merge structure features into the 15m candle DataFrame for storage
             self.candles_15m = pd.merge(self.candles_15m, structure_df, on="bar_time", how="left")

        # Step 6: 4H Market Bias
        self.candles_4h = self._compute_4h_market_bias(self.candles_4h)

        # Step 7: FVG detection (1m)
        fvg_df = self._compute_fvg_smc(self.candles_1m)

        # Step 8: Merge everything to tick resolution
        enriched = self._merge_smc_base(ticks_df, self.candles_1m, self.candles_15m)
        enriched = self._merge_session_levels(enriched, session_levels_df)
        enriched = self._merge_swing_counts(enriched, swing_count_df)
        enriched = self._merge_smc_structure(enriched, structure_df, self.candles_4h)
        enriched = self._merge_fvg_smc(enriched, fvg_df, self.candles_1m)

        # Step 9: Contextual metadata
        enriched = self._add_session_label(enriched)

        logger.info(
            f"[FeatureEngineer] Built SMC features: {len(enriched)} ticks, "
            f"{len(self.candles_15m)} 15-min bars, {len(self.swing_highs_15m)} confirmed 15m swings"
        )
        return enriched

    def save_to_duckdb(self, df: pd.DataFrame, store: "DuckDBStore") -> None:
        """Persist enriched tick DataFrame to the Gold layer DuckDB store."""
        if df.empty:
            return
        cols = [
            "timestamp_utc", "symbol", "bid", "ask", "mid", "volume", "volume_usd", "source",
            "atr_20_1m", "atr_15_15m",
            "prev_day_high", "prev_day_low",
            "current_session_high", "current_session_low",
            "prev_session_high", "prev_session_low", "session_boundary",
            "n_confirmed_swing_highs_15m", "n_confirmed_swing_lows_15m",
            "smc_trend_15m", "hh_15m", "ll_15m", "strong_low_15m", "strong_high_15m",
            "bos_detected_15m", "choch_detected_15m", "market_bias_4h",
            "fvg_high", "fvg_low", "fvg_side", "fvg_filled", "fvg_age_bars", "session"
        ]
        available = [c for c in cols if c in df.columns]
        store.upsert_features(df[available])

        # Save candle tables
        if not self.candles_1m.empty:
            store.upsert_candles("candles_1m", self.candles_1m)
        if not self.candles_15m.empty:
            store.upsert_candles("candles_15m", self.candles_15m)
        if not self.candles_4h.empty:
            store.upsert_candles("candles_4h", self.candles_4h)

    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _resample_ohlc(self, ticks_df: pd.DataFrame, interval: str) -> pd.DataFrame:
        if ticks_df.empty: return pd.DataFrame()
        resampled = ticks_df.set_index("timestamp_utc")["mid"].resample(interval).ohlc()
        resampled = resampled.dropna()
        resampled = resampled.reset_index().rename(columns={
            "timestamp_utc": "bar_time",
            "open": "bar_open", "high": "bar_high", "low": "bar_low", "close": "bar_close"
        })
        return resampled

    def _compute_smc_atr(self, candles: pd.DataFrame, period: int, col: str) -> pd.DataFrame:
        candles = candles.copy()
        if len(candles) < period:
            candles[col] = np.nan
            return candles
        candles[col] = AverageTrueRange(
            high=candles["bar_high"], low=candles["bar_low"], close=candles["bar_close"], window=period
        ).average_true_range()
        return candles

    def _build_session_levels(self, ticks_df: pd.DataFrame, candles_4h: pd.DataFrame) -> pd.DataFrame:
        """
        Walk forward through ticks and snapshot liquidity levels at each
        session boundary (daily midnight + session opens).
        """
        ticks_df = ticks_df.copy().sort_values("timestamp_utc")
        snapshots: list[dict] = []

        # Running state
        prev_day_high:    float = np.nan
        prev_day_low:     float = np.nan
        prev_session_high: float = np.nan
        prev_session_low:  float = np.nan
        cur_session_high:  float = np.nan
        cur_session_low:   float = np.nan

        last_day: int = -1
        last_session: str = ""

        # Determine boundaries from ticks
        for ts, group in ticks_df.groupby(ticks_df["timestamp_utc"].dt.floor("1min")):
            hour = ts.hour
            day  = ts.dayofyear
            
            # 1. Daily Reset (UTC Midnight)
            if day != last_day:
                if last_day != -1:
                    # Capture previous day high/low
                    prev_day_high = cur_session_high
                    prev_day_low  = cur_session_low
                last_day = day

            # 2. Session Boundary Check
            current_session = "asian"
            if hour >= SESSION_BOUNDARIES["ny"]: current_session = "ny"
            elif hour >= SESSION_BOUNDARIES["london"]: current_session = "london"
            
            if current_session != last_session:
                # Capture boundary event
                snapshots.append({
                    "snapshot_time":        ts,
                    "session_boundary":     True,
                    "session_name":         current_session,
                    "prev_day_high":        prev_day_high,
                    "prev_day_low":         prev_day_low,
                    "current_session_high": group["mid"].max(),
                    "current_session_low":  group["mid"].min(),
                    "prev_session_high":    prev_session_high,
                    "prev_session_low":     prev_session_low
                })
                # Reset session trackers
                prev_session_high = cur_session_high
                prev_session_low  = cur_session_low
                cur_session_high  = group["mid"].max()
                cur_session_low   = group["mid"].min()
                last_session      = current_session
            else:
                cur_session_high = max(cur_session_high, group["mid"].max()) if not np.isnan(cur_session_high) else group["mid"].max()
                cur_session_low  = min(cur_session_low, group["mid"].min()) if not np.isnan(cur_session_low) else group["mid"].min()

        return pd.DataFrame(snapshots)

    def _build_swing_history_15m(self, candles_15m: pd.DataFrame) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        highs = candles_15m["bar_high"].values
        lows = candles_15m["bar_low"].values
        # Harden timestamps: strip then localize back to UTC to be safe
        times = pd.to_datetime(candles_15m["bar_time"], utc=True)
        
        sh, sl = [], []
        for i in range(5, len(candles_15m) - 5):
            if highs[i] == highs[i-5:i+6].max():
                sh.append(SwingPoint(float(highs[i]), times[i], "high", i))
            if lows[i] == lows[i-5:i+6].min():
                sl.append(SwingPoint(float(lows[i]), times[i], "low", i))
        return sh, sl

    def _build_swing_count_series(
        self, ticks_df: pd.DataFrame, candles_15m: pd.DataFrame, 
        swing_highs: list[SwingPoint], swing_lows: list[SwingPoint]
    ) -> pd.DataFrame:
        """Build cumulative counts of confirmed swings at each point in time."""
        if ticks_df.empty: return pd.DataFrame()
        
        events = []
        for s in swing_highs: events.append({"time": s.bar_time, "type": "high"})
        for s in swing_lows:  events.append({"time": s.bar_time, "type": "low"})
        
        event_df = pd.DataFrame(events).sort_values("time")
        if event_df.empty: return pd.DataFrame()
        
        event_df["n_confirmed_swing_highs_15m"] = (event_df["type"] == "high").cumsum()
        event_df["n_confirmed_swing_lows_15m"]  = (event_df["type"] == "low").cumsum()
        
        return event_df[["time", "n_confirmed_swing_highs_15m", "n_confirmed_swing_lows_15m"]]

    def _compute_smc_structure_nodes(
        self, candles_15m: pd.DataFrame, swing_highs: list[SwingPoint], swing_lows: list[SwingPoint]
    ) -> pd.DataFrame:
        """Structural Node State Machine (15m). HH, LL, StrongHigh, StrongLow, CHoCH, BOS."""
        if candles_15m.empty: return pd.DataFrame()
        
        candles = candles_15m.copy().sort_values("bar_time")
        n = len(candles)
        
        # State variables
        trend, hh, ll, strong_low, strong_high = None, None, None, None, None
        
        trends, hhs, lls, s_lows, s_highs = [None]*n, [None]*n, [None]*n, [None]*n, [None]*n
        bos_ev, choch_ev = [False]*n, [False]*n
        
        all_swings = sorted(swing_highs + swing_lows, key=lambda s: s.bar_time)
        sw_idx = 0
        closes = candles["bar_close"].values
        # Harden: guaranteed UTC Timestamps
        times = pd.to_datetime(candles["bar_time"], utc=True)

        for i in range(n):
            curr_t, curr_c = times[i], closes[i]
            while sw_idx < len(all_swings) and all_swings[sw_idx].bar_time <= curr_t:
                s = all_swings[sw_idx]
                if s.kind == "high":
                    if trend == "bull" and (hh is None or s.price > hh):
                        hh = s.price
                        pre = [sl for sl in swing_lows if sl.bar_time < s.bar_time]
                        if pre: strong_low = pre[-1].price
                    elif trend is None: hh = s.price
                else: # low
                    if trend == "bear" and (ll is None or s.price < ll):
                        ll = s.price
                        pre = [sh for sh in swing_highs if sh.bar_time < s.bar_time]
                        if pre: strong_high = pre[-1].price
                    elif trend is None: ll = s.price
                sw_idx += 1

            if trend == "bull" or trend is None:
                if hh is not None and curr_c > hh:
                    if trend == "bull": bos_ev[i] = True
                    trend = "bull"
                elif strong_low is not None and curr_c < strong_low:
                    choch_ev[i], trend, ll = True, "bear", curr_c
                    pre = [sh for sh in swing_highs if sh.bar_time < curr_t]
                    strong_high = max(p.price for p in pre) if pre else None
                    hh, strong_low = None, None
            elif trend == "bear":
                if ll is not None and curr_c < ll:
                    bos_ev[i] = True
                elif strong_high is not None and curr_c > strong_high:
                    choch_ev[i], trend, hh = True, "bull", curr_c
                    pre = [sl for sl in swing_lows if sl.bar_time < curr_t]
                    strong_low = min(p.price for p in pre) if pre else None
                    ll, strong_high = None, None
            
            trends[i], hhs[i], lls[i], s_lows[i], s_highs[i] = trend, hh, ll, strong_low, strong_high

        return pd.DataFrame({
            "bar_time": times, "smc_trend_15m": trends, "hh_15m": hhs, "ll_15m": lls,
            "strong_low_15m": s_lows, "strong_high_15m": s_highs,
            "bos_detected_15m": bos_ev, "choch_detected_15m": choch_ev
        })

    def _compute_4h_market_bias(self, candles_4h: pd.DataFrame) -> pd.DataFrame:
        """4H Trend using fixed L=5, R=5 Fractal points."""
        candles = candles_4h.copy()
        if len(candles) < 11:
            candles["market_bias_4h"] = "neutral"
            return candles
        
        highs, lows = candles["bar_high"].values, candles["bar_low"].values
        n, bias = len(candles), ["neutral"]*len(candles)
        last_hh, last_ll, curr_bias = None, None, "neutral"
        
        for i in range(5, n-5):
            if highs[i] == highs[i-5:i+6].max():
                if last_hh is None or highs[i] > last_hh: curr_bias = "bullish"
                last_hh = highs[i]
            if lows[i] == lows[i-5:i+6].min():
                if last_ll is None or lows[i] < last_ll: curr_bias = "bearish"
                last_ll = lows[i]
            bias[i+5] = curr_bias
        
        candles["market_bias_4h"] = bias
        return candles

    def _compute_fvg_smc(self, candles_1m: pd.DataFrame) -> pd.DataFrame:
        n = len(candles_1m)
        if n < 3: return pd.DataFrame()
        highs, lows = candles_1m["bar_high"].values, candles_1m["bar_low"].values
        opens, closes = candles_1m["bar_open"].values, candles_1m["bar_close"].values
        times, atrs = candles_1m["bar_time"].values, candles_1m.get("atr_20_1m", [np.nan]*n)
        
        records = []
        for i in range(1, n-1):
            # Bullish
            if lows[i+1] > highs[i-1]:
                body = abs(closes[i] - opens[i])
                atr = atrs[i] if not np.isnan(atrs[i]) else self.atr_1m_avg
                if body > atr * DISPLACEMENT_FACTOR and (lows[i+1] - highs[i-1]) > max(atr * FVG_ATR_MULTIPLIER, ABSOLUTE_MIN_PIPS):
                    records.append({"formed_at": times[i], "fvg_high": lows[i+1], "fvg_low": highs[i-1], "fvg_side": "bullish_fvg", "fvg_filled": False})
            # Bearish
            if highs[i+1] < lows[i-1]:
                body = abs(closes[i] - opens[i])
                atr = atrs[i] if not np.isnan(atrs[i]) else self.atr_1m_avg
                if body > atr * DISPLACEMENT_FACTOR and (lows[i-1] - highs[i+1]) > max(atr * FVG_ATR_MULTIPLIER, ABSOLUTE_MIN_PIPS):
                    records.append({"formed_at": times[i], "fvg_high": lows[i-1], "fvg_low": highs[i+1], "fvg_side": "bearish_fvg", "fvg_filled": False})
        return pd.DataFrame(records)

    # ── Merge Helpers ──

    def _merge_smc_base(self, ticks_df, c1m, c15m):
        enriched = pd.merge_asof(
            ticks_df.sort_values("timestamp_utc"),
            c1m[["bar_time", "bar_open", "bar_high", "bar_low", "bar_close", "atr_20_1m"]].sort_values("bar_time"),
            left_on="timestamp_utc", right_on="bar_time", direction="backward"
        ).drop(columns="bar_time")
        return enriched

    def _merge_session_levels(self, df: pd.DataFrame, levels: pd.DataFrame) -> pd.DataFrame:
        if levels.empty:
            for c in ["prev_day_high", "prev_day_low", "current_session_high", "current_session_low", "session_boundary"]:
                df[c] = np.nan
            df["session_boundary"] = False
            return df
        return pd.merge_asof(
            df.sort_values("timestamp_utc"),
            levels.sort_values("snapshot_time"),
            left_on="timestamp_utc", right_on="snapshot_time", direction="backward"
        ).drop(columns="snapshot_time", errors="ignore")

    def _merge_swing_counts(self, df: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
        if counts.empty:
            df["n_confirmed_swing_highs_15m"] = 0
            df["n_confirmed_swing_lows_15m"]  = 0
            return df
        return pd.merge_asof(
            df.sort_values("timestamp_utc"),
            counts.sort_values("time"),
            left_on="timestamp_utc", right_on="time", direction="backward"
        ).drop(columns="time", errors="ignore")

    def _merge_smc_structure(self, df, struct, c4h):
        if not struct.empty:
            df = pd.merge_asof(
                df.sort_values("timestamp_utc"),
                struct.sort_values("bar_time"),
                left_on="timestamp_utc", right_on="bar_time", direction="backward"
            ).drop(columns="bar_time", errors="ignore")
        
        if not c4h.empty:
            df = pd.merge_asof(
                df.sort_values("timestamp_utc"),
                c4h[["bar_time", "market_bias_4h"]].sort_values("bar_time"),
                left_on="timestamp_utc", right_on="bar_time", direction="backward"
            ).drop(columns="bar_time", errors="ignore")
        return df

    def _merge_fvg_smc(self, enriched: pd.DataFrame, fvg_df: pd.DataFrame, candles_1m: pd.DataFrame) -> pd.DataFrame:
        if fvg_df.empty:
            for col in ["fvg_high", "fvg_low", "fvg_side", "fvg_filled", "fvg_age_bars"]:
                enriched[col] = None
            return enriched

        enriched = enriched.copy()
        f_highs, f_lows, f_sides = fvg_df["fvg_high"].values, fvg_df["fvg_low"].values, fvg_df["fvg_side"].values
        f_times, f_filled = pd.to_datetime(fvg_df["formed_at"].values, utc=True), fvg_df["fvg_filled"].values
        b_times = pd.to_datetime(candles_1m["bar_time"].values, utc=True)

        o_high, o_low, o_side, o_filled, o_age = [], [], [], [], []

        for _, row in enriched.iterrows():
            mid, t_time, trend = row["mid"], row["timestamp_utc"], row.get("smc_trend_15m")
            valid = np.ones(len(fvg_df), dtype=bool)
            for k in range(len(fvg_df)):
                if f_filled[k] or f_times[k] >= t_time: valid[k] = False; continue
                if (trend == "bull" and f_sides[k] != "bullish_fvg") or (trend == "bear" and f_sides[k] != "bearish_fvg"):
                    valid[k] = False; continue
                
                f_idx = np.searchsorted(b_times, f_times[k], side="right")
                t_idx = np.searchsorted(b_times, t_time, side="right")
                if (t_idx - f_idx) > MAX_FVG_AGE_BARS: valid[k] = False
            
            idx = np.where(valid)[0]
            if len(idx) == 0:
                for o in [o_high, o_low, o_side, o_filled, o_age]: o.append(None)
                continue
            
            best = idx[np.argmin(np.abs((f_highs[idx] + f_lows[idx])/2.0 - mid))]
            o_high.append(f_highs[best]); o_low.append(f_lows[best]); o_side.append(f_sides[best])
            o_filled.append(False); o_age.append(int(np.searchsorted(b_times, t_time, side="right") - np.searchsorted(b_times, f_times[best], side="right")))

        enriched["fvg_high"], enriched["fvg_low"], enriched["fvg_side"] = o_high, o_low, o_side
        enriched["fvg_filled"], enriched["fvg_age_bars"] = o_filled, o_age
        return enriched

    def _add_session_label(self, df: pd.DataFrame) -> pd.DataFrame:
        def label(ts):
            h = ts.hour
            for s, (start, end) in SESSIONS.items():
                if start <= h < end: return s
            return "asian"
        df["session"] = df["timestamp_utc"].apply(label)
        return df
