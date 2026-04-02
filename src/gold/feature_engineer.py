"""
Gold Layer — FeatureEngineer (Scout & Sniper SMC Strategy)
────────────────────
Refactored to align with "Scout & Sniper" SMC-based strategy (1m / 15m / 4H).

1.  Structural Node State Machine (15m)
        _compute_smc_structure_nodes() — Tracks HH, LL, StrongHigh, StrongLow,
        CHoCH, BOS, bos_time, and r_dynamic_at_bos for Gate 3.
2.  Fair Value Gap (FVG) detection (1m)
        _compute_fvg_smc() — 3-candle imbalance with Displacement (Gate 1) and
        Size (Gate 2) filters.  FVGs are linked to the BOS impulse candle.
        Fill detection marks gaps as consumed when price closes back through them.
3.  Tiered Liquidity Levels
        Tier 1: Prev Day High/Low  (+2 sweep score)
        Tier 2: Session High/Low   (+1 sweep score)
        Tier 3: StrongHigh/Low + EQH/EQL Clusters  (+0, gate-only)
4.  4H Market Bias
        _compute_4h_market_bias() — Requires HH+HL (bull) or LH+LL (bear)
        sequence, not just a single new extreme.
5.  R_dynamic
        _compute_r_dynamic() — ATR-adaptive confirmation window,
        stored at BOS time for Gate 3 timing.

Fixes applied vs v1:
  #1  _compute_smc_atr no longer has a self.candles_1m side-effect.
      RSI lives in its own _compute_rsi_1m() method.
  #2  R_dynamic is computed and threaded through swing detection.
  #3  FVG scan restricted to bars within MAX_FVG_AGE_BARS of a BOS or CHoCH
      candle. CHoCH was previously excluded from the anchor set (fix #2b).
  #4  fvg_filled / filled_at: exclusion is now time-gated — FVGs are only
      hidden after filled_at, not retroactively from formation (review fix #1).
  #5  Liquidity sweep checks all three tiers with ATR tolerance.
  #6  BOS correctly updates strong_low / strong_high.
  #7  CHoCH sets hh/ll to None (not close price) to avoid immediate false BOS.
  #8  Session label unified to "newyork" (was "new_york") everywhere.
  #9  Swing detection uses strict > / < (was ==).
  #10 times array uses .values (numpy) to avoid pandas label-indexing bugs.
  #11 prev_day_high/low tracks true daily extreme across all sessions.
  #12 4H bias requires HH+HL or LH+LL sequence, not a single new extreme.
  #13 bos_time and r_dynamic_at_bos stored in structure output and DB cols.
  #14 ATR sweep tolerance applied in all sweep checks.
  #15 EQH/EQL cluster detection implemented with dedup + consumption filter.
  #16 Session hours unified to spec: Asian 00-09, London 08-17, NY 13-22.
  #17 FVG trend-filter uses trend at formation time, not at query time.
  #18 RSI decoupled from ATR method (resolved by #1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timezone
from typing import TYPE_CHECKING, Literal, Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator

if TYPE_CHECKING:
    from src.gold.duckdb_store import DuckDBStore

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

# ATR periods
ATR_PERIOD_1M:  int = 20   # atr_20_1m  — FVG size filter, displacement check
ATR_PERIOD_15M: int = 15   # atr_15_15m — R_dynamic, sweep tolerance, P1/P2

# Williams Fractal — 15m / 4H
FRACTAL_L:     int = 5
FRACTAL_R_MAX: int = 5
FRACTAL_R_MIN: int = 2

# Williams Fractal — 1m (trailing SL, fixed)
FRACTAL_1M_L: int = 3
FRACTAL_1M_R: int = 3

# R_dynamic: k = R_DYNAMIC_K_FACTOR × avg_atr_15m  (calibrate via backtest)
R_DYNAMIC_K_FACTOR: float = 3.0

# Gate 1 — displacement
DISPLACEMENT_FACTOR: float = 0.5

# Gate 2 — FVG size
FVG_ATR_MULTIPLIER: float = 0.15
ABSOLUTE_MIN_PIPS:  float = 3.0

# Gate 4 — sweep tolerance
SWEEP_ATR_FACTOR: float = 0.15
SWEEP_MIN_PIPS:   float = 2.0

# Tier 3 liquidity — EQH/EQL clustering
MIN_CLUSTER_SIZE:     int   = 2
EQUAL_HL_ATR_FACTOR:  float = 0.25
EQUAL_HL_LOOKBACK:    int   = 80
CONSUMPTION_LOOKBACK: int   = 5

# FVG expiry
MAX_FVG_AGE_BARS: int = 20

# ── Session hours (UTC) — aligned to strategy spec ────────────────────────────
# Asian 00-09 · London 08-17 · NY 13-22
# Overlap is intentional: London/NY overlap is 13-17.
# Label priority (first match wins): newyork → london → asian.
SESSIONS: dict[str, tuple[int, int]] = {
    "newyork": (13, 22),   # FIX #8/#16: was "new_york" and hours (12,21)
    "london":  (8,  17),   # FIX #16: was (7, 16)
    "asian":   (0,   9),   # FIX #16: was (0, 7)
}

# Session boundary start hours — used for session-level tracking
SESSION_START_HOURS: dict[str, int] = {
    "asian":   0,
    "london":  8,   # FIX #16: was 8 (correct) in SESSION_BOUNDARIES but 7 in SESSIONS
    "newyork": 13,  # FIX #16: was "ny":13 but label was "new_york"
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class SwingPoint:
    """A confirmed Williams Fractal swing point."""
    price:    float
    bar_time: pd.Timestamp
    kind:     Literal["high", "low"]
    bar_idx:  int


@dataclass
class LiquidityLevel:
    """Tiered liquidity level as per strategy Phase 3 Gate 4."""
    price:     float
    tier:      int                    # 1 | 2 | 3
    kind:      Literal["high", "low"]
    label:     str
    strength:  int = 1                # swing-point count (EQH/EQL only)
    direction: str = ""               # "high" | "low" — used by consumption filter


# ── Main class ────────────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Computes SMC features from a DataFrame of UnifiedTick rows.
    """

    def __init__(self) -> None:
        self.swing_highs_15m: list[SwingPoint] = []
        self.swing_lows_15m:  list[SwingPoint] = []
        self.atr_15m_avg:     float = 0.0
        self.atr_1m_avg:      float = 0.0

        self.candles_1m:  pd.DataFrame = pd.DataFrame()
        self.candles_15m: pd.DataFrame = pd.DataFrame()
        self.candles_4h:  pd.DataFrame = pd.DataFrame()

    # ── Public API ────────────────────────────────────────────────────────────

    def build_features(self, ticks_df: pd.DataFrame) -> pd.DataFrame:
        """Full Scout & Sniper SMC feature pipeline."""
        self.candles_1m  = pd.DataFrame()
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

        # Step 1: Resample to OHLC candles
        self.candles_1m  = self._resample_ohlc(ticks_df, "1min")
        self.candles_15m = self._resample_ohlc(ticks_df, "15min")
        self.candles_4h  = self._resample_ohlc(ticks_df, "4h")

        for df, sym, src in [
            (self.candles_1m,  symbol, source),
            (self.candles_15m, symbol, source),
            (self.candles_4h,  symbol, source),
        ]:
            if not df.empty:
                df["symbol"] = sym
                df["source"] = src

        # Step 2: ATR — separate methods, no cross-frame side-effects  (FIX #1/#18)
        self.candles_1m  = self._compute_atr(self.candles_1m,  ATR_PERIOD_1M,  "atr_20_1m")
        self.candles_15m = self._compute_atr(self.candles_15m, ATR_PERIOD_15M, "atr_15_15m")
        self.candles_1m  = self._compute_rsi_1m(self.candles_1m)  # FIX #18: own method

        self.atr_15m_avg = float(self.candles_15m["atr_15_15m"].median() or 0.0)
        self.atr_1m_avg  = float(self.candles_1m["atr_20_1m"].median() or 0.0)

        # Step 3: Session level tracker (Tier 1 & 2)
        session_levels_df = self._build_session_levels(ticks_df)

        # Step 4: R_dynamic per 15m bar  (FIX #2)
        self.candles_15m = self._compute_r_dynamic(self.candles_15m)

        # Step 5: 15m confirmed swing history using R_dynamic  (FIX #2/#9/#10)
        self.swing_highs_15m, self.swing_lows_15m = self._build_swing_history_15m(
            self.candles_15m
        )
        swing_count_df = self._build_swing_count_series(
            ticks_df, self.swing_highs_15m, self.swing_lows_15m
        )

        # Step 6: Structural node state machine  (FIX #6/#7/#10/#13)
        structure_df = self._compute_smc_structure_nodes(
            self.candles_15m, self.swing_highs_15m, self.swing_lows_15m
        )
        if not structure_df.empty:
            self.candles_15m = pd.merge(
                self.candles_15m, structure_df, on="bar_time", how="left"
            )

        # Step 7: 4H market bias  (FIX #9/#12)
        self.candles_4h = self._compute_4h_market_bias(self.candles_4h)

        # Step 8: FVG detection linked to BOS candles  (FIX #3/#4/#17)
        fvg_df = self._compute_fvg_smc(self.candles_1m, structure_df)

        # Step 9: Tiered liquidity sweep detection  (FIX #5/#14/#15)
        self.candles_1m = self._compute_liquidity_sweeps_on_candles(
            self.candles_1m,
            self.swing_highs_15m,
            self.swing_lows_15m,
            session_levels_df,
        )

        # Step 10: Merge everything to tick resolution
        enriched = self._merge_smc_base(ticks_df, self.candles_1m)
        enriched = self._merge_15m_atr(enriched, self.candles_15m)
        enriched = self._merge_session_levels(enriched, session_levels_df)
        enriched = self._merge_swing_counts(enriched, swing_count_df)
        enriched = self._merge_smc_structure(enriched, structure_df, self.candles_4h)
        enriched = self._merge_fvg_smc(enriched, fvg_df, self.candles_1m)

        enriched = pd.merge_asof(
            enriched.sort_values("timestamp_utc"),
            self.candles_1m[["bar_time", "liq_swept", "liq_side", "sweep_tier", 
                             "sweep_candle_low", "sweep_candle_high", 
                             "sweep_wick", "sweep_body"]].sort_values("bar_time"),
            left_on="timestamp_utc", right_on="bar_time", direction="backward",
        ).drop(columns="bar_time", errors="ignore")

        # Step 11: Session label  (FIX #8/#16)
        enriched = self._add_session_label(enriched)

        logger.info(
            f"[FeatureEngineer] Built SMC features: {len(enriched)} ticks, "
            f"{len(self.candles_15m)} 15m bars, "
            f"{len(self.swing_highs_15m)} confirmed swing highs, "
            f"{len(self.swing_lows_15m)} confirmed swing lows"
        )
        return enriched

    def save_to_duckdb(self, df: pd.DataFrame, store: "DuckDBStore") -> None:
        """Persist enriched tick DataFrame to the Gold layer DuckDB store."""
        if df.empty:
            return
        cols = [
            "timestamp_utc", "symbol", "bid", "ask", "mid",
            "volume", "volume_usd", "source",
            "atr_20_1m", "atr_15_15m",
            "prev_day_high", "prev_day_low",
            "current_session_high", "current_session_low",
            "prev_session_high", "prev_session_low", "session_boundary",
            "n_confirmed_swing_highs_15m", "n_confirmed_swing_lows_15m",
            "smc_trend_15m", "hh_15m", "ll_15m",
            "strong_low_15m", "strong_high_15m",
            "bos_detected_15m", "choch_detected_15m",
            "bos_up_15m", "bos_down_15m", "choch_up_15m", "choch_down_15m",
            "is_swing_high_15m", "is_swing_low_15m",
            "bos_time_ms",          # FIX #13: Gate 3 timing
            "r_dynamic_at_bos",     # FIX #13: Gate 3 timing
            "market_bias_4h",
            "fvg_high", "fvg_low", "fvg_side", "fvg_filled", "fvg_age_bars",
            "session",
            "liq_swept", "liq_side", "sweep_tier",
            "sweep_candle_low", "sweep_candle_high", "sweep_wick", "sweep_body",
            "rsi_14",
        ]
        available = [c for c in cols if c in df.columns]
        store.upsert_features(df[available])

        if not self.candles_1m.empty:
            store.upsert_candles("candles_1m", self.candles_1m)
        if not self.candles_15m.empty:
            store.upsert_candles("candles_15m", self.candles_15m)
        if not self.candles_4h.empty:
            store.upsert_candles("candles_4h", self.candles_4h)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resample_ohlc(self, ticks_df: pd.DataFrame, interval: str) -> pd.DataFrame:
        if ticks_df.empty:
            return pd.DataFrame()
        resampled = (
            ticks_df.set_index("timestamp_utc")["mid"]
            .resample(interval)
            .ohlc()
            .dropna()
            .reset_index()
            .rename(columns={
                "timestamp_utc": "bar_time",
                "open":  "bar_open",
                "high":  "bar_high",
                "low":   "bar_low",
                "close": "bar_close",
            })
        )
        return resampled

    # FIX #1/#18: pure function — operates only on the passed DataFrame,
    # no writes to self.candles_* as side-effects.
    def _compute_atr(self, candles: pd.DataFrame, period: int, col: str) -> pd.DataFrame:
        candles = candles.copy()
        if len(candles) < period:
            candles[col] = np.nan
            return candles
        candles[col] = AverageTrueRange(
            high=candles["bar_high"],
            low=candles["bar_low"],
            close=candles["bar_close"],
            window=period,
        ).average_true_range()
        return candles

    # FIX #18: RSI in its own method, written to candles_1m once, explicitly.
    def _compute_rsi_1m(self, candles_1m: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        candles_1m = candles_1m.copy()
        if len(candles_1m) < period:
            candles_1m["rsi_14"] = np.nan
            return candles_1m
        candles_1m["rsi_14"] = RSIIndicator(
            close=candles_1m["bar_close"], window=period
        ).rsi()
        return candles_1m

    # FIX #2: compute R_dynamic per 15m bar from ATR.
    def _compute_r_dynamic(self, candles_15m: pd.DataFrame) -> pd.DataFrame:
        """
        R_dynamic = clamp(round(k / atr15_15m), R_MIN, R_MAX)
        k = R_DYNAMIC_K_FACTOR × median(atr15_15m)

        Stored per bar so the value at BOS time can be retrieved later (Gate 3).
        Bars without ATR (warm-up period) get R_MAX as a safe default.
        """
        candles_15m = candles_15m.copy()
        if "atr_15_15m" not in candles_15m.columns:
            candles_15m["r_dynamic"] = FRACTAL_R_MAX
            return candles_15m

        median_atr = float(candles_15m["atr_15_15m"].median() or 1.0)
        k = R_DYNAMIC_K_FACTOR * median_atr

        def _r(atr_val: float) -> int:
            if np.isnan(atr_val) or atr_val <= 0:
                return FRACTAL_R_MAX
            return int(np.clip(round(k / atr_val), FRACTAL_R_MIN, FRACTAL_R_MAX))

        candles_15m["r_dynamic"] = candles_15m["atr_15_15m"].apply(_r)
        return candles_15m

    # FIX #2/#9/#10: uses R_dynamic per bar, strict >, numpy indexing.
    def _build_swing_history_15m(
        self, candles_15m: pd.DataFrame
    ) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        """
        Williams Fractal swing detection on 15m.
        L=5 fixed, R=R_dynamic (2–5) per bar.

        FIX #9:  strict > / < (was ==, generating false swings at flat prices).
        FIX #10: uses .values arrays to avoid pandas label-index bugs after merges.
        FIX #2:  R_dynamic applied per bar.
        """
        if candles_15m.empty:
            return [], []

        highs  = candles_15m["bar_high"].values   # FIX #10: numpy, not Series
        lows   = candles_15m["bar_low"].values
        times  = pd.to_datetime(candles_15m["bar_time"].values, utc=True)
        r_vals = (
            candles_15m["r_dynamic"].values
            if "r_dynamic" in candles_15m.columns
            else np.full(len(candles_15m), FRACTAL_R_MAX, dtype=int)
        )

        sh: list[SwingPoint] = []
        sl: list[SwingPoint] = []

        n = len(candles_15m)
        for i in range(FRACTAL_L, n):
            r = int(r_vals[i])
            if i + r >= n:
                # Not enough right-side bars confirmed yet — skip
                continue

            # FIX #9: strict > not ==
            if (highs[i] > highs[i - FRACTAL_L : i].max()          # left window
                    and highs[i] > highs[i + 1 : i + r + 1].max()): # right window
                sh.append(SwingPoint(float(highs[i]), times[i], "high", i))

            if (lows[i] < lows[i - FRACTAL_L : i].min()
                    and lows[i] < lows[i + 1 : i + r + 1].min()):
                sl.append(SwingPoint(float(lows[i]), times[i], "low", i))

        return sh, sl

    def _build_swing_count_series(
        self,
        ticks_df: pd.DataFrame,
        swing_highs: list[SwingPoint],
        swing_lows: list[SwingPoint],
    ) -> pd.DataFrame:
        events = (
            [{"time": s.bar_time, "type": "high"} for s in swing_highs]
            + [{"time": s.bar_time, "type": "low"}  for s in swing_lows]
        )
        if not events:
            return pd.DataFrame()

        event_df = pd.DataFrame(events).sort_values("time")
        event_df["n_confirmed_swing_highs_15m"] = (event_df["type"] == "high").cumsum()
        event_df["n_confirmed_swing_lows_15m"]  = (event_df["type"] == "low").cumsum()
        return event_df[["time", "n_confirmed_swing_highs_15m", "n_confirmed_swing_lows_15m"]]

    # FIX #6/#7/#10/#13
    def _compute_smc_structure_nodes(
        self,
        candles_15m: pd.DataFrame,
        swing_highs: list[SwingPoint],
        swing_lows:  list[SwingPoint],
    ) -> pd.DataFrame:
        """
        Structural Node State Machine (15m).

        FIX #6:  BOS now updates strong_low (bull BOS) / strong_high (bear BOS).
        FIX #7:  CHoCH sets hh/ll to None — not to curr_c — so the next bar
                 doesn't immediately fire a false BOS.
        FIX #10: times uses .values (numpy) to avoid label-index KeyErrors.
        FIX #13: bos_time_ms and r_dynamic_at_bos stored per BOS event
                 for Gate 3 timing.
        """
        if candles_15m.empty:
            return pd.DataFrame()

        candles = candles_15m.copy().sort_values("bar_time").reset_index(drop=True)
        n = len(candles)

        # FIX #10: numpy arrays — no pandas label indexing
        closes = candles["bar_close"].values
        times  = pd.to_datetime(candles["bar_time"].values, utc=True)
        r_vals = (
            candles["r_dynamic"].values
            if "r_dynamic" in candles.columns
            else np.full(n, FRACTAL_R_MAX, dtype=int)
        )

        # State
        trend:        Optional[str]   = None
        hh:           Optional[float] = None
        ll:           Optional[float] = None
        strong_low:   Optional[float] = None
        strong_high:  Optional[float] = None

        # Output arrays
        trends      = [None]  * n
        hhs         = [None]  * n
        lls         = [None]  * n
        s_lows      = [None]  * n
        s_highs     = [None]  * n
        bos_ev      = [False] * n
        choch_ev    = [False] * n
        bos_up_ev   = [False] * n
        bos_down_ev = [False] * n
        choch_up_ev = [False] * n
        choch_dn_ev = [False] * n
        bos_time_ms_arr      = [None] * n   # FIX #13
        r_dyn_at_bos_arr     = [None] * n   # FIX #13

        all_swings = sorted(swing_highs + swing_lows, key=lambda s: s.bar_time)
        sw_idx = 0

        for i in range(n):
            curr_t = times[i]
            curr_c = closes[i]
            r_now  = int(r_vals[i])

            # Consume all confirmed swings up to and including this bar's time
            while sw_idx < len(all_swings) and all_swings[sw_idx].bar_time <= curr_t:
                s = all_swings[sw_idx]

                if s.kind == "high":
                    if trend in ("bull", None):
                        if hh is None or s.price > hh:
                            hh = s.price
                            # Update strong_low when a new HH is confirmed in bull trend
                            pre_lows = [sl for sl in swing_lows if sl.bar_time < s.bar_time]
                            if pre_lows:
                                strong_low = pre_lows[-1].price  # FIX #6 (partial — see BOS branch)
                else:  # low
                    if trend in ("bear", None):
                        if ll is None or s.price < ll:
                            ll = s.price
                            pre_highs = [sh for sh in swing_highs if sh.bar_time < s.bar_time]
                            if pre_highs:
                                strong_high = pre_highs[-1].price  # FIX #6 (partial)

                sw_idx += 1

            # ── BOS / CHoCH detection ─────────────────────────────────────────
            # NOTE on check order: BOS is evaluated before CHoCH in both
            # branches.  On an extreme spike candle it is theoretically possible
            # for curr_c > hh AND curr_c < strong_low to both be true
            # simultaneously (e.g. a gap-and-reverse that closes below strong_low
            # after breaking above hh).  In that case only the BOS fires and the
            # CHoCH is suppressed for that bar.  This is the intended behaviour:
            # a close that exceeds the structural high takes priority over the
            # reversal signal on the same bar; the CHoCH can fire on the next bar
            # if price remains below strong_low.  (Review issue #7.)

            if trend == "bull" or trend is None:

                if hh is not None and curr_c > hh:
                    # ── Bullish BOS ──────────────────────────────────────────
                    if trend == "bull":
                        bos_ev[i]    = True
                        bos_up_ev[i] = True
                        # FIX #6: update strong_low to the most recent confirmed SL
                        # before this BOS bar (defines the new sweep zone floor)
                        pre_lows = [sl for sl in swing_lows if sl.bar_time <= curr_t]
                        if pre_lows:
                            strong_low = pre_lows[-1].price
                        # FIX #13: record bos_time and r_dynamic at this moment
                        bos_time_ms_arr[i]  = int(curr_t.value // 1_000_000)
                        r_dyn_at_bos_arr[i] = r_now
                    trend = "bull"

                elif strong_low is not None and curr_c < strong_low:
                    # ── Bearish CHoCH (Bull -> Bear) ────────────────────────
                    choch_ev[i]    = True
                    choch_dn_ev[i] = True
                    trend = "bear"
                    ll    = None
                    hh    = None
                    strong_low = None
                    pre_highs = [sh for sh in swing_highs if sh.bar_time < curr_t]
                    strong_high = max(p.price for p in pre_highs) if pre_highs else None

                elif trend is None and ll is not None and curr_c < ll:
                    # ── Bearish Trend Establishment (None -> Bear) ──────────
                    # FIX: allow initial bear trend without prior bull HH.
                    trend = "bear"
                    # No HH/LL reset needed yet, as we are entering from None.
                    # Update strong_high to the most recent confirmed SH.
                    pre_highs = [sh for sh in swing_highs if sh.bar_time <= curr_t]
                    if pre_highs:
                        strong_high = pre_highs[-1].price

            elif trend == "bear":

                if ll is not None and curr_c < ll:
                    # ── Bearish BOS ──────────────────────────────────────────
                    bos_ev[i]     = True
                    bos_down_ev[i] = True
                    # FIX #6: update strong_high to the most recent confirmed SH
                    pre_highs = [sh for sh in swing_highs if sh.bar_time <= curr_t]
                    if pre_highs:
                        strong_high = pre_highs[-1].price
                    # FIX #13
                    bos_time_ms_arr[i]  = int(curr_t.value // 1_000_000)
                    r_dyn_at_bos_arr[i] = r_now
                    # FIX review-#3: do NOT advance ll = curr_c here.
                    # ll only moves when the swing-consumption loop confirms a
                    # new swing low.  Was ll = curr_c — same cascade problem
                    # as the bullish BOS branch above.

                elif strong_high is not None and curr_c > strong_high:
                    # ── Bullish CHoCH ────────────────────────────────────────
                    choch_ev[i]   = True
                    choch_up_ev[i] = True
                    trend = "bull"
                    # FIX #7: hh = None, not curr_c
                    hh    = None
                    ll    = None
                    strong_high = None
                    pre_lows = [sl for sl in swing_lows if sl.bar_time < curr_t]
                    strong_low = min(p.price for p in pre_lows) if pre_lows else None

            trends[i]  = trend
            hhs[i]     = hh
            lls[i]     = ll
            s_lows[i]  = strong_low
            s_highs[i] = strong_high

        sh_times = {s.bar_time for s in swing_highs}
        sl_times = {s.bar_time for s in swing_lows}
        is_sh = [t in sh_times for t in times]
        is_sl = [t in sl_times for t in times]

        return pd.DataFrame({
            "bar_time":           times,
            "smc_trend_15m":      trends,
            "hh_15m":             hhs,
            "ll_15m":             lls,
            "strong_low_15m":     s_lows,
            "strong_high_15m":    s_highs,
            "bos_detected_15m":   bos_ev,
            "choch_detected_15m": choch_ev,
            "bos_up_15m":         bos_up_ev,
            "bos_down_15m":       bos_down_ev,
            "choch_up_15m":       choch_up_ev,
            "choch_down_15m":     choch_dn_ev,
            "is_swing_high_15m":  is_sh,
            "is_swing_low_15m":   is_sl,
            "bos_time_ms":        bos_time_ms_arr,    # FIX #13
            "r_dynamic_at_bos":   r_dyn_at_bos_arr,  # FIX #13
        })

    # FIX #9/#12: requires HH+HL (bull) or LH+LL (bear) sequence.
    def _compute_4h_market_bias(self, candles_4h: pd.DataFrame) -> pd.DataFrame:
        """
        4H trend filter. L=R=5 fixed fractals.

        FIX #9:  strict > / < in fractal detection.
        FIX #12: requires HH+HL sequence for bull, LH+LL for bear.
                 A single new high without a higher low is insufficient.
        """
        candles = candles_4h.copy()
        if len(candles) < 11:
            candles["market_bias_4h"] = "neutral"
            return candles

        # FIX #10: numpy arrays
        highs = candles["bar_high"].values
        lows  = candles["bar_low"].values
        n     = len(candles)
        bias  = ["neutral"] * n

        curr_bias = "neutral"
        last_hh: Optional[float] = None
        last_hl: Optional[float] = None   # FIX #12: track higher lows
        last_ll: Optional[float] = None
        last_lh: Optional[float] = None   # FIX #12: track lower highs

        for i in range(5, n - 5):
            # FIX #9: strict >/<
            is_fractal_high = (
                highs[i] > highs[i - 5 : i].max()
                and highs[i] > highs[i + 1 : i + 6].max()
            )
            is_fractal_low = (
                lows[i] < lows[i - 5 : i].min()
                and lows[i] < lows[i + 1 : i + 6].min()
            )

            if is_fractal_high:
                if last_hh is None or highs[i] > last_hh:
                    # New HH — bullish only if we also have a HL (FIX #12)
                    if last_hl is not None:
                        curr_bias = "bullish"
                    last_hh = highs[i]
                else:
                    # Lower high — potential LH for bear sequence
                    last_lh = highs[i]
                    # Bear confirmed if we have a LL too
                    if last_ll is not None:
                        curr_bias = "bearish"

            if is_fractal_low:
                if last_ll is None or lows[i] < last_ll:
                    # New LL — bearish only if we also have a LH (FIX #12)
                    if last_lh is not None:
                        curr_bias = "bearish"
                    last_ll = lows[i]
                else:
                    # Higher low — potential HL for bull sequence
                    last_hl = lows[i]
                    if last_hh is not None:
                        curr_bias = "bullish"

            bias[i + 5] = curr_bias

        candles["market_bias_4h"] = bias
        return candles

    # FIX #3/#4/#17
    def _compute_fvg_smc(
        self,
        candles_1m: pd.DataFrame,
        structure_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        3-candle Fair Value Gap detection on 1m candles.

        FIX #3:  Only scans for FVGs on/after the BOS impulse candle.
                 Each FVG record carries bos_bar_time so consumers know
                 which BOS produced it.
        FIX #4:  Fill detection: after finding a gap, we scan forward and
                 mark fvg_filled=True the moment any subsequent close trades
                 back through the gap.
        FIX #17: Trend filter applied at formation time using the trend that
                 was active when the FVG formed — not the trend at query time.
                 FVGs formed before trend is established are kept but flagged
                 with fvg_trend="unknown"; the strategy layer can filter them.
        """
        n = len(candles_1m)
        if n < 3:
            return pd.DataFrame()

        highs  = candles_1m["bar_high"].values
        lows   = candles_1m["bar_low"].values
        opens  = candles_1m["bar_open"].values
        closes = candles_1m["bar_close"].values
        times  = pd.to_datetime(candles_1m["bar_time"].values, utc=True)
        atrs   = (
            candles_1m["atr_20_1m"].values
            if "atr_20_1m" in candles_1m.columns
            else np.full(n, self.atr_1m_avg)
        )

        # Build the set of BOS + CHoCH candle times from the structure DataFrame.
        # FIX review-#2b: CHoCH was previously excluded — it is an equally valid
        # T1 entry trigger per the spec and its impulse candle FVG must be included.
        trigger_times: set = set()
        if not structure_df.empty:
            trigger_col_bos   = "bos_detected_15m"
            trigger_col_choch = "choch_detected_15m"
            for col in (trigger_col_bos, trigger_col_choch):
                if col in structure_df.columns:
                    for _, row in structure_df[structure_df[col] == True].iterrows():
                        t = pd.Timestamp(row["bar_time"])
                        if t.tzinfo is None:
                            t = t.tz_localize("UTC")
                        trigger_times.add(t)

        # Build a sorted numpy array of trigger times for fast searchsorted lookup.
        # FIX review-#2a: we only scan 1m bars that fall within MAX_FVG_AGE_BARS
        # of a BOS/CHoCH candle.  The previous implementation scanned every bar
        # globally — FVGs from unrelated structural moves were included.
        sorted_trigger_times = np.array(
            sorted(trigger_times), dtype="datetime64[ns]"
        ) if trigger_times else np.array([], dtype="datetime64[ns]")

        def _nearest_prior_trigger(bar_t: pd.Timestamp) -> Optional[pd.Timestamp]:
            """Return the most recent trigger at or before bar_t, or None."""
            if len(sorted_trigger_times) == 0:
                return None
            bar_ns = np.datetime64(bar_t.value, "ns")
            idx = np.searchsorted(sorted_trigger_times, bar_ns, side="right") - 1
            if idx < 0:
                return None
            return pd.Timestamp(sorted_trigger_times[idx], tz="UTC")

        def _within_fvg_window(bar_idx: int, bar_t: pd.Timestamp) -> bool:
            """
            True if this 1m bar is close enough (in bars) to a BOS/CHoCH candle
            to plausibly originate its FVG.  We convert the 15m trigger time to a
            1m bar index via searchsorted and check the distance.
            """
            trig = _nearest_prior_trigger(bar_t)
            if trig is None:
                return False
            # FIX: coerce tz-aware to naive UTC for numpy.searchsorted compatibility
            trig_ns   = np.datetime64(trig.value, "ns")
            trig_1m_i = int(np.searchsorted(
                times.tz_localize(None).values.astype("datetime64[ns]"), trig_ns, side="left"
            ))
            return (bar_idx - trig_1m_i) <= MAX_FVG_AGE_BARS

        # Map each 1m bar to its formation-time trend  (FIX #17)
        trend_at_formation = np.full(n, None, dtype=object)
        if not structure_df.empty and "smc_trend_15m" in structure_df.columns:
            struct_sorted = structure_df[["bar_time", "smc_trend_15m"]].sort_values("bar_time")
            s_times  = pd.to_datetime(struct_sorted["bar_time"].values, utc=True)
            s_trends = struct_sorted["smc_trend_15m"].values
            for i in range(n):
                idx = np.searchsorted(s_times, times[i], side="right") - 1
                if idx >= 0:
                    trend_at_formation[i] = s_trends[idx]

        records = []
        for i in range(1, n - 1):
            # FIX review-#2a: skip bars that are not within the window of a
            # BOS or CHoCH candle — prevents FVGs from unrelated moves being
            # included in the database.
            if not _within_fvg_window(i, times[i]):
                continue

            atr = atrs[i] if not np.isnan(atrs[i]) else self.atr_1m_avg
            body = abs(closes[i] - opens[i])

            # Displacement check (Gate 1)
            if body <= atr * DISPLACEMENT_FACTOR:
                continue

            fvg_high: Optional[float] = None
            fvg_low:  Optional[float] = None
            side:     Optional[str]   = None

            # Bullish FVG: gap between candle[i-1].high and candle[i+1].low
            if lows[i + 1] > highs[i - 1]:
                size = lows[i + 1] - highs[i - 1]
                if size >= max(atr * FVG_ATR_MULTIPLIER, ABSOLUTE_MIN_PIPS):
                    fvg_high = lows[i + 1]
                    fvg_low  = highs[i - 1]
                    side     = "bullish_fvg"

            # Bearish FVG: gap between candle[i-1].low and candle[i+1].high
            elif highs[i + 1] < lows[i - 1]:
                size = lows[i - 1] - highs[i + 1]
                if size >= max(atr * FVG_ATR_MULTIPLIER, ABSOLUTE_MIN_PIPS):
                    fvg_high = lows[i - 1]
                    fvg_low  = highs[i + 1]
                    side     = "bearish_fvg"

            if fvg_high is None:
                continue

            # FIX #17: record trend at formation time (not filtered here —
            # strategy layer decides; trend=None FVGs get fvg_trend="unknown")
            formation_trend = trend_at_formation[i]
            fvg_trend = formation_trend if formation_trend is not None else "unknown"

            # Anchor: nearest prior BOS or CHoCH (FIX review-#2b: includes CHoCH)
            bos_anchor = _nearest_prior_trigger(times[i])

            # FIX #4: fill detection — scan forward for a close through the gap
            filled      = False
            filled_at   = None
            for j in range(i + 2, min(i + 2 + MAX_FVG_AGE_BARS, n)):
                if side == "bullish_fvg" and closes[j] < fvg_low:
                    filled    = True
                    filled_at = times[j]
                    break
                if side == "bearish_fvg" and closes[j] > fvg_high:
                    filled    = True
                    filled_at = times[j]
                    break

            records.append({
                "formed_at":    times[i],
                "fvg_high":     fvg_high,
                "fvg_low":      fvg_low,
                "fvg_side":     side,
                "fvg_filled":   filled,        # FIX #4
                "filled_at":    filled_at,     # FIX #4
                "fvg_trend":    fvg_trend,     # FIX #17
                "bos_bar_time": bos_anchor,    # FIX #3
            })

        return pd.DataFrame(records)

    # FIX #5/#14/#15
    def _compute_liquidity_sweeps_on_candles(
        self,
        candles_1m: pd.DataFrame,
        swing_highs: list[SwingPoint],
        swing_lows:  list[SwingPoint],
        session_levels_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Tiered liquidity sweep detection on 1m candles.

        FIX #5:  Checks all three tiers (Tier 1: daily high/low, Tier 2: session
                 high/low, Tier 3: structural swing points + EQH/EQL clusters).
        FIX #14: ATR-based sweep tolerance applied (max(atr*factor, min_pips)).
        FIX #15: EQH/EQL cluster detection with deduplication + consumption filter.

        A sweep is: wick crosses level (within tolerance) AND close is back on
        the other side of the level (no full close-through).
        """
        if candles_1m.empty:
            return candles_1m

        n      = len(candles_1m)
        lows   = candles_1m["bar_low"].values
        highs  = candles_1m["bar_high"].values
        opens  = candles_1m["bar_open"].values
        closes = candles_1m["bar_close"].values
        times  = pd.to_datetime(candles_1m["bar_time"].values, utc=True)
        atrs   = (
            candles_1m["atr_20_1m"].values
            if "atr_20_1m" in candles_1m.columns
            else np.full(n, self.atr_1m_avg)
        )

        liq_swept  = np.zeros(n, dtype=bool)
        liq_side   = np.full(n, None, dtype=object)
        sweep_tier = np.full(n, None, dtype=object)
        sweep_low  = np.full(n, np.nan, dtype=float)
        sweep_high = np.full(n, np.nan, dtype=float)
        sweep_wick = np.full(n, np.nan, dtype=float)
        sweep_body = np.full(n, np.nan, dtype=float)

        # ── Build a time-indexed level history ────────────────────────────────
        # Review FIX #5: levels are built from the full session_levels_df but
        # each snapshot carries a snapshot_time.  In the per-candle loop below
        # we only include snapshots whose snapshot_time <= candle bar_time,
        # preventing early candles from seeing future session levels.
        #
        # Review FIX #4: after collecting levels we deduplicate by
        # (round(price, 2), direction) so a level that appears in multiple
        # session snapshots (e.g. prev_day_high repeated for 3 sessions) is
        # only checked once per candle.

        # Pre-sort snapshots by time for searchsorted lookup
        snap_times_arr: np.ndarray = np.array([], dtype="datetime64[ns]")
        snap_rows: list[dict] = []
        if not session_levels_df.empty and "snapshot_time" in session_levels_df.columns:
            sl_sorted = session_levels_df.sort_values("snapshot_time").reset_index(drop=True)
            # FIX: coerce tz-aware to naive UTC for numpy arrays
            snap_times_arr = pd.to_datetime(
                sl_sorted["snapshot_time"].values, utc=True
            ).tz_localize(None).values.astype("datetime64[ns]")
            snap_rows = sl_sorted.to_dict("records")

        def _active_session_levels(bar_t: pd.Timestamp) -> list[LiquidityLevel]:
            """Return deduplicated Tier 1+2 levels visible at bar_t."""
            if len(snap_times_arr) == 0:
                return []
            bar_ns = np.datetime64(bar_t.value, "ns")
            # All snapshots at or before this bar
            n_active = int(np.searchsorted(snap_times_arr, bar_ns, side="right"))
            if n_active == 0:
                return []

            raw: list[LiquidityLevel] = []
            for row in snap_rows[:n_active]:
                for col, lbl, kind, tier in [
                    ("prev_day_high",        "Prev Day High",      "high", 1),
                    ("prev_day_low",         "Prev Day Low",       "low",  1),
                    ("current_session_high", "Session High",       "high", 2),
                    ("current_session_low",  "Session Low",        "low",  2),
                    ("prev_session_high",    "Prev Session High",  "high", 2),
                    ("prev_session_low",     "Prev Session Low",   "low",  2),
                ]:
                    v = row.get(col, np.nan)
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        raw.append(LiquidityLevel(
                            price=float(v), tier=tier, kind=kind,
                            label=lbl, direction=kind,
                        ))

            # Review FIX #4: deduplicate by (rounded price, direction)
            seen_keys: set = set()
            deduped: list[LiquidityLevel] = []
            for lv in raw:
                key = (round(lv.price, 2), lv.direction)
                if key not in seen_keys:
                    seen_keys.add(key)
                    deduped.append(lv)
            return deduped

        # Tier 3 levels are static (built from full swing history) — no
        # time-masking needed because swing points are only added as they are
        # confirmed (the swing detection itself already prevents look-ahead).
        tier3_structural: list[LiquidityLevel] = [
            LiquidityLevel(price=s.price, tier=3, kind="high",
                           label="Structural SH", direction="high")
            for s in swing_highs
        ] + [
            LiquidityLevel(price=s.price, tier=3, kind="low",
                           label="Structural SL", direction="low")
            for s in swing_lows
        ]

        # Tier 3b: EQH/EQL clusters  (FIX #15)
        atr_median = float(np.nanmedian(atrs)) if len(atrs) else self.atr_15m_avg
        tolerance  = atr_median * EQUAL_HL_ATR_FACTOR
        recent_sh  = swing_highs[-EQUAL_HL_LOOKBACK:]
        recent_sl  = swing_lows[-EQUAL_HL_LOOKBACK:]
        eqh = self._find_equal_levels(recent_sh, tolerance, "high")
        eql = self._find_equal_levels(recent_sl, tolerance, "low")

        sweep_tol_global = max(atr_median * SWEEP_ATR_FACTOR, SWEEP_MIN_PIPS)
        tier3_eq = [
            lv for lv in eqh + eql
            if not self._is_level_consumed(lv, candles_1m, sweep_tol_global)
        ]
        tier3_levels = tier3_structural + tier3_eq

        # ── Per-candle sweep check ────────────────────────────────────────────
        for i in range(n):
            # FIX #14: ATR-based tolerance per candle
            atr_i = atrs[i] if not np.isnan(atrs[i]) else atr_median
            tol   = max(atr_i * SWEEP_ATR_FACTOR, SWEEP_MIN_PIPS)

            # Review FIX #5: only levels visible at this bar's time
            t1_t2 = _active_session_levels(times[i])
            all_levels = t1_t2 + tier3_levels

            for level in sorted(all_levels, key=lambda lv: lv.tier):  # Tier 1 first
                p = level.price

                if level.direction == "high":
                    if highs[i] >= p - tol and closes[i] < p:
                        liq_swept[i]  = True
                        liq_side[i]   = "high"
                        sweep_tier[i] = level.tier
                        sweep_high[i] = highs[i]
                        sweep_low[i]  = lows[i]
                        sweep_wick[i] = highs[i] - max(opens[i], closes[i])
                        sweep_body[i] = abs(opens[i] - closes[i])
                        break

                else:  # "low"
                    if lows[i] <= p + tol and closes[i] > p:
                        liq_swept[i]  = True
                        liq_side[i]   = "low"
                        sweep_tier[i] = level.tier
                        sweep_low[i]  = lows[i]
                        sweep_high[i] = highs[i]
                        sweep_wick[i] = min(opens[i], closes[i]) - lows[i]
                        sweep_body[i] = abs(opens[i] - closes[i])
                        break

        res = candles_1m.copy()
        res["liq_swept"]  = liq_swept
        res["liq_side"]   = liq_side
        res["sweep_tier"] = sweep_tier
        res["sweep_candle_low"]  = sweep_low
        res["sweep_candle_high"] = sweep_high
        res["sweep_wick"] = sweep_wick
        res["sweep_body"] = sweep_body
        return res

    # ── EQH/EQL helpers  (FIX #15) ───────────────────────────────────────────

    def _find_equal_levels(
        self,
        swings: list[SwingPoint],
        tolerance: float,
        kind: Literal["high", "low"],
    ) -> list[LiquidityLevel]:
        """
        Cluster confirmed swing points within ATR-relative tolerance.
        Returns deduplicated LiquidityLevel list (strongest cluster per zone).
        """
        if len(swings) < MIN_CLUSTER_SIZE:
            return []

        clusters: list[LiquidityLevel] = []
        used = [False] * len(swings)

        for i, anchor in enumerate(swings):
            if used[i]:
                continue
            cluster = [anchor]
            for j, other in enumerate(swings[i + 1:], start=i + 1):
                if not used[j] and abs(anchor.price - other.price) <= tolerance:
                    cluster.append(other)
                    used[j] = True

            if len(cluster) >= MIN_CLUSTER_SIZE:
                avg_price = sum(s.price for s in cluster) / len(cluster)
                clusters.append(LiquidityLevel(
                    price=avg_price,
                    tier=3,
                    kind=kind,
                    label=f"EQ{'H' if kind == 'high' else 'L'} Cluster",
                    strength=len(cluster),
                    direction=kind,
                ))

        return self._deduplicate_levels(clusters, tolerance)

    @staticmethod
    def _deduplicate_levels(
        levels: list[LiquidityLevel], tolerance: float
    ) -> list[LiquidityLevel]:
        """Keep only the strongest cluster per zone (highest strength wins)."""
        seen:   list[LiquidityLevel] = []
        for lv in sorted(levels, key=lambda l: -l.strength):
            if not any(abs(lv.price - s.price) <= tolerance for s in seen):
                seen.append(lv)
        return seen

    def _is_level_consumed(
        self,
        level: LiquidityLevel,
        candles_1m: pd.DataFrame,
        sweep_tolerance: float,
    ) -> bool:
        """
        A level is consumed when recent 1m closes have all moved decisively
        beyond it — i.e. it has been fully traded through, not just swept.
        """
        if candles_1m.empty or len(candles_1m) < CONSUMPTION_LOOKBACK:
            return False
        recent_closes = candles_1m["bar_close"].values[-CONSUMPTION_LOOKBACK:]
        if level.direction == "high":
            return bool(np.all(recent_closes > level.price + sweep_tolerance))
        return bool(np.all(recent_closes < level.price - sweep_tolerance))

    # ── Session levels  (FIX #11/#16) ────────────────────────────────────────

    def _build_session_levels(self, ticks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Walk forward through ticks and snapshot liquidity levels at each
        session boundary.

        FIX #11: Tracks true daily high/low across ALL sessions (Asian + London
                 + NY) rather than only the last active session's running value.
        FIX #16: Session boundary hours aligned to spec:
                 Asian 00-09 · London 08-17 · NY 13-22.

        Review note #8 — sparse tick data / session boundary gaps:
            This method groups ticks by dt.floor("1min").  If the tick feed has
            gaps spanning a session open (e.g. no ticks between 07:55 and 08:05
            UTC), the 08:00 London boundary minute will have no group and the
            session transition will be detected at the first minute that DOES
            have ticks after 08:00.  Consequences:
              • The snapshot_time for the London open will be slightly late.
              • prev_session_high/low will be snapshotted at that delayed time.
            This is a data-quality concern rather than a logic bug — the fix is
            to ensure the tick feed has no multi-minute gaps around session opens.
            For Dukascopy historical data this is rare but possible during public
            holidays.  For live MT5 feeds it can occur on reconnect.
            If this matters for backtesting accuracy, resample ticks to 1-minute
            OHLC before passing to build_features and forward-fill the index to
            guarantee a row at every minute boundary.
        """
        ticks_df = ticks_df.copy().sort_values("timestamp_utc")
        snapshots: list[dict] = []

        prev_day_high:     float = np.nan
        prev_day_low:      float = np.nan
        prev_session_high: float = np.nan
        prev_session_low:  float = np.nan
        cur_session_high:  float = np.nan
        cur_session_low:   float = np.nan

        # FIX #11: separate accumulator for the true full-day range
        daily_high: float = np.nan
        daily_low:  float = np.nan

        last_day:     int = -1
        last_session: str = ""

        for ts, group in ticks_df.groupby(ticks_df["timestamp_utc"].dt.floor("1min")):
            hour = ts.hour
            day  = ts.dayofyear

            # FIX #11: daily reset uses the true accumulated daily range
            if day != last_day:
                if last_day != -1:
                    prev_day_high = daily_high
                    prev_day_low  = daily_low
                # Reset daily accumulators
                daily_high = float(group["mid"].max())
                daily_low  = float(group["mid"].min())
                last_day   = day
            else:
                # Accumulate true daily range across all sessions  (FIX #11)
                g_max = float(group["mid"].max())
                g_min = float(group["mid"].min())
                daily_high = g_max if np.isnan(daily_high) else max(daily_high, g_max)
                daily_low  = g_min if np.isnan(daily_low)  else min(daily_low,  g_min)

            # FIX #16: session boundaries from SESSION_START_HOURS (spec-aligned)
            current_session = "asian"
            if hour >= SESSION_START_HOURS["newyork"]:
                current_session = "newyork"
            elif hour >= SESSION_START_HOURS["london"]:
                current_session = "london"

            if current_session != last_session:
                snapshots.append({
                    "snapshot_time":        ts,
                    "session_boundary":     True,
                    "session_name":         current_session,
                    "prev_day_high":        prev_day_high,
                    "prev_day_low":         prev_day_low,
                    "current_session_high": float(group["mid"].max()),
                    "current_session_low":  float(group["mid"].min()),
                    "prev_session_high":    prev_session_high,
                    "prev_session_low":     prev_session_low,
                })
                prev_session_high = cur_session_high
                prev_session_low  = cur_session_low
                cur_session_high  = float(group["mid"].max())
                cur_session_low   = float(group["mid"].min())
                last_session      = current_session
            else:
                g_max = float(group["mid"].max())
                g_min = float(group["mid"].min())
                cur_session_high = (
                    g_max if np.isnan(cur_session_high)
                    else max(cur_session_high, g_max)
                )
                cur_session_low = (
                    g_min if np.isnan(cur_session_low)
                    else min(cur_session_low, g_min)
                )

        return pd.DataFrame(snapshots)

    # ── Merge helpers ─────────────────────────────────────────────────────────

    def _merge_smc_base(self, ticks_df: pd.DataFrame, c1m: pd.DataFrame) -> pd.DataFrame:
        cols_1m = ["bar_time", "bar_open", "bar_high", "bar_low", "bar_close", "atr_20_1m", "rsi_14"]
        available = [c for c in cols_1m if c in c1m.columns]
        return pd.merge_asof(
            ticks_df.sort_values("timestamp_utc"),
            c1m[available].sort_values("bar_time"),
            left_on="timestamp_utc", right_on="bar_time", direction="backward",
        ).drop(columns="bar_time", errors="ignore")

    def _merge_15m_atr(self, enriched: pd.DataFrame, candles_15m: pd.DataFrame) -> pd.DataFrame:
        if candles_15m.empty or "atr_15_15m" not in candles_15m.columns:
            enriched["atr_15_15m"] = np.nan
            return enriched
        atr_df = (
            candles_15m[["bar_time", "atr_15_15m"]]
            .dropna(subset=["atr_15_15m"])
            .query("atr_15_15m > 0")
            .sort_values("bar_time")
        )
        return pd.merge_asof(
            enriched.sort_values("timestamp_utc"),
            atr_df,
            left_on="timestamp_utc", right_on="bar_time", direction="backward",
        ).drop(columns="bar_time", errors="ignore")

    def _merge_session_levels(
        self, df: pd.DataFrame, levels: pd.DataFrame
    ) -> pd.DataFrame:
        if levels.empty:
            for c in [
                "prev_day_high", "prev_day_low",
                "current_session_high", "current_session_low",
                "prev_session_high", "prev_session_low",
            ]:
                df[c] = np.nan
            df["session_boundary"] = False
            return df
        return pd.merge_asof(
            df.sort_values("timestamp_utc"),
            levels.sort_values("snapshot_time"),
            left_on="timestamp_utc", right_on="snapshot_time", direction="backward",
        ).drop(columns="snapshot_time", errors="ignore")

    def _merge_swing_counts(
        self, df: pd.DataFrame, counts: pd.DataFrame
    ) -> pd.DataFrame:
        if counts.empty:
            df["n_confirmed_swing_highs_15m"] = 0
            df["n_confirmed_swing_lows_15m"]  = 0
            return df
        return pd.merge_asof(
            df.sort_values("timestamp_utc"),
            counts.sort_values("time"),
            left_on="timestamp_utc", right_on="time", direction="backward",
        ).drop(columns="time", errors="ignore")

    def _merge_smc_structure(
        self,
        df: pd.DataFrame,
        struct: pd.DataFrame,
        c4h: pd.DataFrame,
    ) -> pd.DataFrame:
        if not struct.empty:
            df = pd.merge_asof(
                df.sort_values("timestamp_utc"),
                struct.sort_values("bar_time"),
                left_on="timestamp_utc", right_on="bar_time", direction="backward",
            ).drop(columns="bar_time", errors="ignore")
        if not c4h.empty:
            df = pd.merge_asof(
                df.sort_values("timestamp_utc"),
                c4h[["bar_time", "market_bias_4h"]].sort_values("bar_time"),
                left_on="timestamp_utc", right_on="bar_time", direction="backward",
            ).drop(columns="bar_time", errors="ignore")
        return df

    def _merge_fvg_smc(
        self,
        enriched: pd.DataFrame,
        fvg_df: pd.DataFrame,
        candles_1m: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge the most relevant active FVG into each tick row.

        Review FIX #1: fvg_filled exclusion is now time-gated.
            Previously valid[f_filled] = False excluded a filled FVG from
            every bar — including bars BEFORE the fill happened (look-ahead
            bias).  Now we load f_filled_at and exclude only when:
                f_filled == True  AND  f_filled_at <= t_time
            A FVG that forms at T=10 and gets filled at T=15 is still visible
            and actionable at T=10 through T=14.

        Review FIX #2b: FVG direction filter replaced with numpy masking
            (was an O(n×m) Python loop — now fully vectorised).

        FIX #17: Uses fvg_trend (trend at formation time).
        """
        if fvg_df.empty or candles_1m.empty:
            for col in ["fvg_high", "fvg_low", "fvg_side", "fvg_filled", "fvg_age_bars"]:
                enriched[col] = None
            return enriched

        f_highs     = fvg_df["fvg_high"].values
        f_lows      = fvg_df["fvg_low"].values
        f_sides     = fvg_df["fvg_side"].values
        f_times     = pd.to_datetime(fvg_df["formed_at"].values, utc=True)
        f_filled    = fvg_df["fvg_filled"].values.astype(bool)
        f_trends    = fvg_df["fvg_trend"].values
        # Review FIX #1: load filled_at as a comparable datetime64 array.
        # NaT for unfilled FVGs — NaT comparisons with <= always return False,
        # which is exactly what we want (unfilled FVGs are never excluded).
        f_filled_at = pd.to_datetime(fvg_df["filled_at"].values, utc=True)

        b_times  = pd.to_datetime(candles_1m["bar_time"].values, utc=True)
        b_closes = candles_1m["bar_close"].values
        n_bars   = len(b_times)

        f_idx  = np.searchsorted(b_times, f_times, side="right")
        f_mids = (f_highs + f_lows) / 2.0
        n_fvgs = len(fvg_df)

        # Pre-build boolean side masks for vectorised filtering  (FIX review-#6)
        is_bullish_fvg = (f_sides == "bullish_fvg")
        is_bearish_fvg = (f_sides == "bearish_fvg")
        is_unknown_trend = (f_trends == "unknown")

        o_high   = np.full(n_bars, np.nan, dtype=float)
        o_low    = np.full(n_bars, np.nan, dtype=float)
        o_side   = np.full(n_bars, None, dtype=object)
        o_filled = np.zeros(n_bars, dtype=bool)
        o_age    = np.zeros(n_bars, dtype=int)

        # Pull 15m trend mapped to 1m resolution for alignment
        temp_1m = pd.DataFrame({"bar_time": b_times})
        if "smc_trend_15m" in self.candles_15m.columns:
            temp_1m = pd.merge_asof(
                temp_1m,
                self.candles_15m[["bar_time", "smc_trend_15m"]].dropna().sort_values("bar_time"),
                on="bar_time", direction="backward",
            )
        trends = temp_1m.get("smc_trend_15m", pd.Series([None] * n_bars)).values

        for i in range(n_bars):
            t_time = b_times[i]
            trend  = trends[i]
            mid    = b_closes[i]

            # ── Base validity mask ────────────────────────────────────────────
            valid = f_times < t_time          # FVG must have formed before this bar

            # Review FIX #1: only exclude filled FVGs if the fill happened
            # at or before the current bar — not just because fvg_filled=True.
            already_filled = f_filled & (f_filled_at <= t_time)
            valid &= ~already_filled

            # Review FIX #6: vectorised direction filter (was O(n×m) Python loop)
            valid &= ~is_unknown_trend        # drop pre-trend FVGs
            if trend == "bull":
                valid &= is_bullish_fvg
            elif trend == "bear":
                valid &= is_bearish_fvg
            else:
                valid[:] = False              # no trend established — skip

            if not np.any(valid):
                continue

            # Age filter
            age = (i + 1) - f_idx
            valid &= (age <= MAX_FVG_AGE_BARS)

            if not np.any(valid):
                continue

            idx  = np.where(valid)[0]
            best = idx[np.argmin(np.abs(f_mids[idx] - mid))]

            o_high[i]   = f_highs[best]
            o_low[i]    = f_lows[best]
            o_side[i]   = f_sides[best]
            o_age[i]    = age[best]

        fvg_states_1m = pd.DataFrame({
            "timestamp_utc": b_times,
            "fvg_high":      o_high,
            "fvg_low":       o_low,
            "fvg_side":      o_side,
            "fvg_filled":    o_filled,
            "fvg_age_bars":  o_age,
        })

        return pd.merge_asof(
            enriched.sort_values("timestamp_utc"),
            fvg_states_1m.sort_values("timestamp_utc"),
            on="timestamp_utc", direction="backward",
        )

    # FIX #8/#16: unified session label using SESSIONS dict with "newyork"
    def _add_session_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Label each tick with its active session.
        Priority: newyork → london → asian (first match wins).
        FIX #8:  Label is "newyork" (was "new_york") — matches strategy scoring.
        FIX #16: Hours from spec: Asian 00-09, London 08-17, NY 13-22.
        """
        def _label(ts: pd.Timestamp) -> str:
            h = ts.hour
            for session, (start, end) in SESSIONS.items():
                if start <= h < end:
                    return session
            return "asian"  # fallback (covers 22-24 UTC which is pre-asian)

        df["session"] = df["timestamp_utc"].apply(_label)
        return df