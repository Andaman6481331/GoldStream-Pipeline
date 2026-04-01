"""
Strategy — Scout & Sniper SMC Decision Engine
XAUUSD | Tick-level execution

Scout & Sniper framework:

    SCOUT phase  — detects valid market structure shift
                   (BOS for continuation, CHoCH for reversal)
                   aligned with 4H market bias.

    SNIPER phase — fires a limit order at the FVG midpoint ONLY IF:
                   • T1 was stopped out at a TRUE LOSS (before Point 1 / breakeven)
                   • All hard gates pass (FVG displacement, size, structure, sweep)
                   • Aggregate confidence score ≥ MIN_SNIPER_SCORE (0–10 scale)

Decision hierarchy:
    1. Spread gate         — skip wide-spread / thin-market ticks
    2. T1 SCOUT entry      — BOS or CHoCH confirmed on 15m, fires freely
                             direction from bos_up/down_15m, choch_up/down_15m
                             4H bias NOT checked — scout probes both directions
    3. T2 SNIPER entry     — only when T1 stopped at a real loss (Point 1 NOT yet hit)
       Gate 1: FVG exists, not refilled, impulse body ≥ atr20_1m × DISPLACEMENT_FACTOR
       Gate 2: FVG size ≥ max(atr20_1m × FVG_ATR_MULTIPLIER, ABSOLUTE_MIN_PIPS)
       Gate 3: 15m structure still intact (time-conditional on r_dynamic at BOS time)
       Gate 4: Liquidity sweep confirmed (tiered levels)
       Scoring: Sweep strength (0–5) + FVG quality (0–5) = 0–10
       Threshold: with-trend vs 4H bias → 4+  |  counter-trend or neutral → 7+

T1 direction logic:
    Direction is determined by the directional signal columns from the feature engineer:
        bos_up_15m   / choch_up_15m   → LONG  (bull BOS / bull CHoCH)
        bos_down_15m / choch_down_15m → SHORT (bear BOS / bear CHoCH)
    These are cross-checked against market_bias_4h — counter-trend signals are ignored.
    bos_direction (relayed by engine) is only used for T2 Gate 3 — it is NEVER
    populated at T1 decision time and must NOT be used to determine T1 direction.

DB columns required in tick_features:
    smc_trend_15m           str   — "bull" | "bear" | "neutral"
    bos_detected_15m        bool  — any BOS confirmed this tick (either direction)
    choch_detected_15m      bool  — any CHoCH confirmed this tick (either direction)
    bos_up_15m              bool  — bullish BOS (close above HH)
    bos_down_15m            bool  — bearish BOS (close below LL)
    choch_up_15m            bool  — bullish CHoCH (bear→bull flip)
    choch_down_15m          bool  — bearish CHoCH (bull→bear flip)
    market_bias_4h          str   — "bullish" | "bearish" | "neutral"
    fvg_high                float — top boundary of nearest active FVG
    fvg_low                 float — bottom boundary of nearest active FVG
    fvg_side                str   — "bullish_fvg" | "bearish_fvg"
    fvg_filled              bool  — True = gap already closed, skip
    fvg_age_bars            int   — closed 1m candles since FVG formed
    fvg_impulse_candle      bool  — True = FVG formed on the BOS impulse candle itself
    fvg_inside_4h_ob        bool  — True = FVG midpoint sits inside a 4H order block
    liq_swept               bool  — True = a liquidity level was swept
    liq_side                str   — "high" | "low" (which side was swept)
    liq_tier                int   — 1 | 2 | 3  (tier of swept level)
    sweep_wick              float — wick size of sweep candle (price units)
    sweep_body              float — body size of sweep candle (price units)
    -- relayed by engine, NOT from DB row:
    bos_direction           str   — "bull" | "bear" — stored at T1 fire time, for T2 Gate 3
    bos_time_ms             int   — unix ms — for T2 Gate 3 timing
    r_dynamic_at_bos        int   — R_dynamic at T1 fire time — for T2 Gate 3
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from src.backtest.backtest_engine import Action
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Context dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScoutSniperContext:
    """
    Full tick context for the Scout & Sniper engine.
    All fields map to tick_features DB columns or are relayed by the engine.
    """

    # ── Identifiers ───────────────────────────────────────────────────────
    timestamp_utc: pd.Timestamp
    symbol:        str

    # ── Price ─────────────────────────────────────────────────────────────
    mid:    float
    bid:    float
    ask:    float
    spread: float

    # ── Session ───────────────────────────────────────────────────────────
    session:        Optional[str]    = None   # "london" | "newyork" | "asian" | "off"
    bar_close:      Optional[float]  = None

    # ── Fair Value Gap (1m) ───────────────────────────────────────────────
    fvg_high:           Optional[float] = None
    fvg_low:            Optional[float] = None
    fvg_side:           Optional[str]   = None   # "bullish_fvg" | "bearish_fvg"
    fvg_filled:         bool            = False
    fvg_age_bars:       Optional[int]   = None
    fvg_impulse_candle: bool            = False
    fvg_inside_4h_ob:   bool            = False

    # ── ATR ───────────────────────────────────────────────────────────────
    atr_20_1m:  Optional[float] = None
    atr_15_15m: Optional[float] = None

    # ── Session levels ────────────────────────────────────────────────────
    prev_day_high:        Optional[float] = None
    prev_day_low:         Optional[float] = None
    current_session_high: Optional[float] = None
    current_session_low:  Optional[float] = None
    prev_session_high:    Optional[float] = None
    prev_session_low:     Optional[float] = None
    session_boundary:     bool            = False

    # ── Swing point counts ────────────────────────────────────────────────
    n_confirmed_swing_highs_15m: int = 0
    n_confirmed_swing_lows_15m:  int = 0

    # ── SMC Phase 2 — Structural Nodes & Bias ─────────────────────────────
    smc_trend_15m:      Optional[str]   = None
    hh_15m:             Optional[float] = None
    ll_15m:             Optional[float] = None
    strong_low_15m:     Optional[float] = None
    strong_high_15m:    Optional[float] = None
    market_bias_4h:     Optional[str]   = None   # "bullish" | "bearish" | "neutral"
    fvg_timestamp:      Optional[pd.Timestamp] = None

    # ── Directional signal breakdown (FROM DB ROW — used for T1 direction) ──
    # These are the correct source of T1 trade direction.
    bos_detected_15m:   bool = False   # any BOS this tick
    choch_detected_15m: bool = False   # any CHoCH this tick
    bos_up_15m:         bool = False   # bullish BOS — triggers T1 LONG
    bos_down_15m:       bool = False   # bearish BOS — triggers T1 SHORT
    choch_up_15m:       bool = False   # bullish CHoCH — triggers T1 LONG
    choch_down_15m:     bool = False   # bearish CHoCH — triggers T1 SHORT
    is_swing_high_15m:  bool = False
    is_swing_low_15m:   bool = False

    # ── Liquidity sweep ───────────────────────────────────────────────────
    liq_swept:   bool            = False
    liq_side:    Optional[str]   = None   # "high" | "low"
    liq_tier:    Optional[int]   = None   # 1 | 2 | 3
    sweep_candle_low:  Optional[float] = None
    sweep_candle_high: Optional[float] = None
    sweep_wick:        Optional[float] = None
    sweep_body:        Optional[float] = None
    rsi_14:            Optional[float] = None

    # ── BOS context — relayed by ENGINE at T1 fire time, for T2 Gate 3 only ──
    # These are NEVER populated from the DB row at T1 decision time.
    # They are passed as kwargs by the engine after T1 has already fired.
    bos_direction:    Optional[str] = None   # "bull" | "bear"
    bos_time_ms:      Optional[int] = None   # unix ms
    r_dynamic_at_bos: Optional[int] = None   # R stored at BOS time

    # ── Trade status (relayed by engine) ──────────────────────────────────
    t1_active:          bool = False
    t2_active:          bool = False
    t1_stopped_at_loss: bool = False   # True ONLY if T1 stopped before Point 1


# ─────────────────────────────────────────────────────────────────────────────
# Thresholds  (tune during backtesting)
# ─────────────────────────────────────────────────────────────────────────────

MAX_SPREAD_ATR_RATIO = 0.25   # skip if spread > 25% of atr_15_15m
MAX_FVG_AGE_BARS     = 10     # discard FVGs older than N closed 1m bars

DISPLACEMENT_FACTOR  = 0.5    # Gate 1: impulse body ≥ atr20_1m × this
FVG_ATR_MULTIPLIER   = 0.15   # Gate 2: FVG size floor fraction of atr20_1m
ABSOLUTE_MIN_PIPS    = 3.0    # Gate 2: FVG absolute size floor (pips)
FVG_WIDTH_SCORE_PIPS = 15.0   # FVG score: width > this → +2
MIN_SNIPER_SCORE     = 4      # minimum 0–10 total score for T2
T2_TIMEOUT_MS        = 10 * 60 * 1000   # 10 min
MIN_SNIPER_SCORE         = 4   # with-trend T2 minimum score
MIN_SNIPER_SCORE_COUNTER = 7   # counter-trend T2 minimum score (higher bar)


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DecisionSummary:
    action:   Action
    reason:   str  = "hold"
    metadata: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def make_decision(ctx: ScoutSniperContext) -> DecisionSummary:
    """
    Core strategy decision logic for Scout & Sniper.

    T1 direction is determined by the directional signal columns:
        bos_up_15m / choch_up_15m   → LONG
        bos_down_15m / choch_down_15m → SHORT
    No 4H bias gate on T1 — scout fires freely in both directions.

    T2 sniper applies bias-aware score threshold:
        - With-trend (bos_direction aligns with market_bias_4h): MIN_SNIPER_SCORE (4)
        - Counter-trend (bos_direction opposes market_bias_4h):  MIN_SNIPER_SCORE_COUNTER (7)
        - Bias neutral: treated as counter-trend (7)
    """

    # ── Gate: spread filter ───────────────────────────────────────────────
    atr_15m = ctx.atr_15_15m
    if atr_15m and atr_15m > 0 and ctx.spread > atr_15m * MAX_SPREAD_ATR_RATIO:
        if ctx.bos_detected_15m:
            print(f"DEBUG HOLD Spread Gate: spread={ctx.spread}, atr={atr_15m}, max={atr_15m*MAX_SPREAD_ATR_RATIO}")
        return DecisionSummary(Action.HOLD, "REASON_SPREAD")

    # ─────────────────────────────────────────────────────────────────────
    # TRADE 1 — SCOUT
    # Fires freely on any confirmed 15m BOS/CHoCH.
    # Direction comes from directional signal columns only.
    # 4H bias is NOT checked here.
    # ─────────────────────────────────────────────────────────────────────
    if not ctx.t1_active and not ctx.t2_active:
        if ctx.bos_detected_15m:
            print(f"DEBUG BOS detected at {ctx.timestamp_utc}: up={ctx.bos_up_15m}, down={ctx.bos_down_15m}, t1_act={ctx.t1_active}, t2_act={ctx.t2_active}")

        if ctx.bos_up_15m or ctx.choch_up_15m:
            reason = "BOS" if ctx.bos_up_15m else "CHoCH"
            res = DecisionSummary(Action.OPEN_T1_LONG, reason)
            if ctx.bos_detected_15m:
                print(f"DEBUG make_decision returning: {res.action.value} reason={res.reason}")
            return res

        if ctx.bos_down_15m or ctx.choch_down_15m:
            reason = "BOS" if ctx.bos_down_15m else "CHoCH"
            res = DecisionSummary(Action.OPEN_T1_SHORT, reason)
            if ctx.bos_detected_15m:
                print(f"DEBUG make_decision returning: {res.action.value} reason={res.reason}")
            return res

    # ─────────────────────────────────────────────────────────────────────
    # TRADE 2 — SNIPER
    # Only when T1 stopped at real loss (before Point 1), T2 not already active.
    # Score threshold is bias-aware: counter-trend requires 7+, with-trend 4+.
    # ─────────────────────────────────────────────────────────────────────
    if not ctx.t1_active and ctx.t1_stopped_at_loss and not ctx.t2_active:

        # ── Hard Gate 1 — FVG exists, not refilled ────────────────────────
        if not ctx.fvg_side or ctx.fvg_filled:
            return DecisionSummary(Action.HOLD, "REASON_NO_FVG")

        if ctx.fvg_high is None or ctx.fvg_low is None:
            return DecisionSummary(Action.HOLD, "REASON_NO_FVG")

        atr_1m = ctx.atr_20_1m
        if atr_1m is None or atr_1m <= 0:
            return DecisionSummary(Action.HOLD, "REASON_NO_ATR")

        # ── Hard Gate 2 — FVG size above noise floor ──────────────────────
        fvg_size = ctx.fvg_high - ctx.fvg_low
        min_fvg  = max(atr_1m * FVG_ATR_MULTIPLIER, ABSOLUTE_MIN_PIPS)
        if fvg_size < min_fvg:
            return DecisionSummary(Action.HOLD, "REASON_FVG_TOO_SMALL")

        # ── Hard Gate 3 — 15m structure still intact (time-conditional) ───
        if ctx.bos_time_ms is not None and ctx.r_dynamic_at_bos is not None:
            now_ms     = int(time.time() * 1000)
            lag_ms     = ctx.r_dynamic_at_bos * 15 * 60 * 1000
            elapsed_ms = now_ms - ctx.bos_time_ms
            if elapsed_ms >= lag_ms:
                if not _is_15m_structure_intact(ctx):
                    return DecisionSummary(Action.HOLD, "REASON_STRUCTURE_BROKEN")

        # ── Hard Gate 4 — Liquidity sweep confirmed ────────────────────────
        if not ctx.liq_swept:
            return DecisionSummary(Action.HOLD, "REASON_NO_LIQUIDITY")

        if ctx.liq_tier is None:
            return DecisionSummary(Action.HOLD, "REASON_NO_LIQ_TIER")

        # ── Sweep Strength Score (0–5) ────────────────────────────────────
        sweep_score = 0
        if ctx.liq_tier == 1:
            sweep_score += 2
        elif ctx.liq_tier == 2:
            sweep_score += 1
        if (
            ctx.sweep_wick is not None
            and ctx.sweep_body is not None
            and ctx.sweep_body > 0
            and ctx.sweep_wick > ctx.sweep_body * 0.5
        ):
            sweep_score += 1
        if ctx.session in ("london", "newyork"):
            sweep_score += 1
        if ctx.choch_detected_15m:
            sweep_score += 1
        sweep_score = min(sweep_score, 5)

        # ── FVG Quality Score (0–5) ───────────────────────────────────────
        fvg_score = 0
        if fvg_size > FVG_WIDTH_SCORE_PIPS:
            fvg_score += 2
        if ctx.fvg_impulse_candle:
            fvg_score += 1
        if ctx.fvg_inside_4h_ob:
            fvg_score += 1
        if ctx.fvg_age_bars is not None and ctx.fvg_age_bars > 3:
            fvg_score += 1
        fvg_score = min(fvg_score, 5)

        total_score = sweep_score + fvg_score   # 0–10

        # ── Bias-aware score threshold ────────────────────────────────────
        # Determine if this T2 is with-trend or counter-trend vs 4H bias.
        # bos_direction is "bull" (long) or "bear" (short), relayed from engine.
        bos_dir = ctx.bos_direction
        bias    = ctx.market_bias_4h   # "bullish" | "bearish" | "neutral" | None

        is_with_trend = (
            (bos_dir == "bull" and bias == "bullish")
            or
            (bos_dir == "bear" and bias == "bearish")
        )
        # Neutral bias or missing bias → treated as counter-trend (stricter threshold)
        required_score = MIN_SNIPER_SCORE if is_with_trend else MIN_SNIPER_SCORE_COUNTER

        if total_score < required_score:
            return DecisionSummary(
                Action.HOLD,
                "REASON_SCORE_TOO_LOW",
                {
                    "total_score":    total_score,
                    "sweep_score":    sweep_score,
                    "fvg_score":      fvg_score,
                    "required_score": required_score,
                    "is_with_trend":  is_with_trend,
                },
            )

        # ── Build limit order metadata ────────────────────────────────────
        fvg_mid   = (ctx.fvg_low + ctx.fvg_high) / 2.0
        now_ms    = int(time.time() * 1000)
        expire_at = now_ms + T2_TIMEOUT_MS

        order_meta = {
            "total_score":    total_score,
            "sweep_score":    sweep_score,
            "fvg_score":      fvg_score,
            "required_score": required_score,
            "is_with_trend":  is_with_trend,
            "fvg_mid":        fvg_mid,
            "fvg_low":        ctx.fvg_low,
            "fvg_high":       ctx.fvg_high,
            "expire_at":      expire_at,
        }

        if (bos_dir == "bull" or ctx.liq_side == "low") and ctx.fvg_side == "bullish_fvg":
            return DecisionSummary(Action.OPEN_T2_LONG, "LIQ_SWEEP", order_meta)

        if (bos_dir == "bear" or ctx.liq_side == "high") and ctx.fvg_side == "bearish_fvg":
            return DecisionSummary(Action.OPEN_T2_SHORT, "LIQ_SWEEP", order_meta)

    return DecisionSummary(Action.HOLD, "hold")

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_15m_structure_intact(ctx: ScoutSniperContext) -> bool:
    """
    Gate 3: uses bos_direction relayed from engine (stored at T1 fire time).
    """
    if ctx.bos_direction == "bull":
        return ctx.hh_15m is not None and ctx.strong_low_15m is not None
    if ctx.bos_direction == "bear":
        return ctx.ll_15m is not None and ctx.strong_high_15m is not None
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Row builder
# ─────────────────────────────────────────────────────────────────────────────

def build_context_from_row(
    row: dict,
    *,
    t1_stopped_at_loss: bool         = False,
    t1_active:          bool         = False,
    t2_active:          bool         = False,
    bos_direction:      Optional[str] = None,   # relayed by engine after T1 fires
    bos_time_ms:        Optional[int] = None,   # relayed by engine after T1 fires
    r_dynamic_at_bos:   Optional[int] = None,   # relayed by engine after T1 fires
) -> ScoutSniperContext:
    """
    Converts a tick_features DB row into a ScoutSniperContext.

    IMPORTANT: bos_direction / bos_time_ms / r_dynamic_at_bos are engine-relay
    fields. They are None at T1 decision time by design. T1 direction is
    determined by bos_up_15m / bos_down_15m / choch_up_15m / choch_down_15m
    which come from the DB row.
    """
    bid = float(row.get("bid", 0.0))
    ask = float(row.get("ask", 0.0))
    mid = (bid + ask) / 2.0

    return ScoutSniperContext(
        timestamp_utc=row.get("timestamp_utc"),
        symbol=row.get("symbol", "XAUUSD"),
        mid=mid,
        bid=bid,
        ask=ask,
        spread=ask - bid,

        session=row.get("session"),
        bar_close=_safe_float(row.get("bar_close")),

        fvg_high=_safe_float(row.get("fvg_high")),
        fvg_low=_safe_float(row.get("fvg_low")),
        fvg_side=row.get("fvg_side"),
        fvg_filled=_safe_bool(row.get("fvg_filled")),
        fvg_age_bars=_safe_int(row.get("fvg_age_bars")),
        fvg_impulse_candle=_safe_bool(row.get("fvg_impulse_candle")),
        fvg_inside_4h_ob=_safe_bool(row.get("fvg_inside_4h_ob")),

        atr_20_1m=_safe_float(row.get("atr_20_1m")),
        atr_15_15m=_safe_float(row.get("atr_15_15m")),

        prev_day_high=_safe_float(row.get("prev_day_high")),
        prev_day_low=_safe_float(row.get("prev_day_low")),
        current_session_high=_safe_float(row.get("current_session_high")),
        current_session_low=_safe_float(row.get("current_session_low")),
        prev_session_high=_safe_float(row.get("prev_session_high")),
        prev_session_low=_safe_float(row.get("prev_session_low")),
        session_boundary=_safe_bool(row.get("session_boundary")),
        n_confirmed_swing_highs_15m=_safe_int(row.get("n_confirmed_swing_highs_15m")) or 0,
        n_confirmed_swing_lows_15m=_safe_int(row.get("n_confirmed_swing_lows_15m")) or 0,

        smc_trend_15m=row.get("smc_trend_15m"),
        hh_15m=_safe_float(row.get("hh_15m")),
        ll_15m=_safe_float(row.get("ll_15m")),
        strong_low_15m=_safe_float(row.get("strong_low_15m")),
        strong_high_15m=_safe_float(row.get("strong_high_15m")),
        market_bias_4h=row.get("market_bias_4h"),
        fvg_timestamp=row.get("fvg_timestamp"),

        # Directional signal columns — source of T1 direction
        bos_detected_15m=_safe_bool(row.get("bos_detected_15m")),
        choch_detected_15m=_safe_bool(row.get("choch_detected_15m")),
        bos_up_15m=_safe_bool(row.get("bos_up_15m")),
        bos_down_15m=_safe_bool(row.get("bos_down_15m")),
        choch_up_15m=_safe_bool(row.get("choch_up_15m")),
        choch_down_15m=_safe_bool(row.get("choch_down_15m")),
        is_swing_high_15m=_safe_bool(row.get("is_swing_high_15m")),
        is_swing_low_15m=_safe_bool(row.get("is_swing_low_15m")),

        liq_swept=_safe_bool(row.get("liq_swept")),
        liq_side=row.get("liq_side"),
        liq_tier=_safe_int(row.get("liq_tier")),
        sweep_candle_low=_safe_float(row.get("sweep_candle_low")),
        sweep_candle_high=_safe_float(row.get("sweep_candle_high")),
        sweep_wick=_safe_float(row.get("sweep_wick")),
        sweep_body=_safe_float(row.get("sweep_body")),

        rsi_14=_safe_float(row.get("rsi_14")),

        # Engine-relay fields — None at T1 time, populated by engine for T2
        bos_direction=bos_direction,
        bos_time_ms=bos_time_ms,
        r_dynamic_at_bos=r_dynamic_at_bos,

        t1_active=t1_active,
        t2_active=t2_active,
        t1_stopped_at_loss=t1_stopped_at_loss,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Type helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(val) -> Optional[float]:
    try:
        import math
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _safe_bool(val) -> bool:
    if val is None:
        return False
    try:
        if pd.isna(val):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes")
    return bool(val)


def _safe_int(val) -> Optional[int]:
    try:
        if val is None:
            return None
        return int(float(val))
    except (TypeError, ValueError):
        return None