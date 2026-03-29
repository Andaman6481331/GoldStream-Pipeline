"""
Strategy — Scout & Sniper SMC Decision Engine
XAUUSD | Tick-level execution

Scout & Sniper framework:

    SCOUT phase  — detects valid market structure shift
                   (BOS for continuation, CHoCH for reversal)

    SNIPER phase — waits for price to retrace into a fresh,
                   unfilled Fair Value Gap (FVG) before firing

Decision hierarchy:
    1. Session gate        — killzone / london only
    2. Spread gate         — skip thin-market ticks
    3. Structure gate      — BOS or CHoCH must be confirmed
    4. FVG gate            — fresh, unfilled, correctly-sided FVG must exist
    5. Entry gate          — price must be INSIDE the FVG (sniper precision)
    6. Premium/discount    — long in discount, short in premium
    7. RSI confirmation    — momentum alignment filter (loose, not primary)

New DB columns required in tick_features (add to FeatureEngineer):
    structure_direction  str   — "bullish" | "bearish"
    bos_detected         bool  — Break of Structure confirmed this tick
    choch_detected       bool  — Change of Character confirmed this tick
    fvg_high             float — top boundary of nearest active FVG
    fvg_low              float — bottom boundary of nearest active FVG
    fvg_side             str   — "bullish_fvg" | "bearish_fvg"
    fvg_filled           bool  — True = gap already closed, skip
    fvg_age_bars         int   — candles since FVG formed (freshness filter)
"""

from __future__ import annotations

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
    All fields map to tick_features DB columns.
    """

    # ── Price ─────────────────────────────────────────────────────────────
    mid:    float
    bid:    float
    ask:    float
    spread: float

    # ── Session & position ────────────────────────────────────────────────
    session:        Optional[str]
    price_position: Optional[str]
    bar_close:      Optional[float]

    # ── Fair Value Gap (1m Sniper entry) ──────────────────────────────────
    fvg_high:     Optional[float] = None
    fvg_low:      Optional[float] = None
    fvg_side:     Optional[str]    = None
    fvg_filled:   bool = False
    fvg_age_bars: Optional[int]    = None

    # ── SMC Phase 1 (Multi-timeframe & Liquidity) ─────────────────────────
    atr_20_1m:    Optional[float] = None
    atr_15_15m:   Optional[float] = None
    prev_day_high: Optional[float] = None
    prev_day_low:  Optional[float] = None
    current_session_high: Optional[float] = None
    current_session_low:  Optional[float] = None
    prev_session_high:    Optional[float] = None
    prev_session_low:     Optional[float] = None
    session_boundary:     bool = False
    n_confirmed_swing_highs_15m: int = 0
    n_confirmed_swing_lows_15m:  int = 0
    rsi_14:                      Optional[float] = None

    # ── SMC Phase 2 (Structural Nodes & Bias) ─────────────────────────────
    smc_trend_15m:      Optional[str]   = None
    hh_15m:             Optional[float] = None
    ll_15m:             Optional[float] = None
    strong_low_15m:     Optional[float] = None
    strong_high_15m:    Optional[float] = None
    bos_detected_15m:   bool = False
    choch_detected_15m: bool = False
    market_bias_4h:     Optional[str]   = None
    fvg_timestamp:      Optional[pd.Timestamp] = None

    # ── Trade status ──────────────────────────────────────────────────────
    t1_active:      bool = False
    t2_active:      bool = False
    t1_stopped_out: bool = False

    # ── Liquidity (Sniper entry) ──────────────────────────────────────────
    liq_swept:   bool = False
    liq_side:    Optional[str] = None # "high" | "low"


# ─────────────────────────────────────────────────────────────────────────────
# Thresholds  (tune during backtesting)
# ─────────────────────────────────────────────────────────────────────────────

MAX_SPREAD_ATR_RATIO = 0.25   # skip if spread > 25% of ATR
MAX_FVG_AGE_BARS     = 20     # discard stale FVGs older than N bars
MIN_SNIPER_SCORE     = 3      # Phase 3: Minimum aggregate score for T2 

# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DecisionSummary:
    action: Action
    reason: str = "hold"  # Descriptive reason for the decision
    metadata: dict = field(default_factory=dict) # Score, BOS type, etc.


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def make_decision(ctx: ScoutSniperContext) -> DecisionSummary:
    """
    Core strategy decision logic for Scout & Sniper.
    Sequential Phase logic: 
    1. Scout (T1) enters on Structure Break (BOS/CHoCH).
    2. Sniper (T2) enters ONLY IF T1 was stopped out at a loss (pnl < 0).
    """

    # ── Gate 1: spread filter (SMC: use 15m ATR) ──────────────────────────
    atr = ctx.atr_15_15m or 0.0
    if atr > 0:
        if ctx.spread > atr * MAX_SPREAD_ATR_RATIO:
            return DecisionSummary(Action.HOLD, "REASON_SPREAD")

    # ── Gate 2: 4H Bias ───────────────────────────────────────────────────
    if not ctx.market_bias_4h or ctx.market_bias_4h == "neutral":
        return DecisionSummary(Action.HOLD, "REASON_BIAS")

    # ── TRADE 1: SCOUT (Structural Break) ─────────────────────────────────
    if not ctx.t1_active and not ctx.t2_active:
        # Long Scout
        if ctx.market_bias_4h == "bullish" and ctx.smc_trend_15m == "bull":
            if ctx.bos_detected_15m:
                return DecisionSummary(Action.OPEN_T1_LONG, "BOS")
            if ctx.choch_detected_15m:
                return DecisionSummary(Action.OPEN_T1_LONG, "CHoCH")
        
        # Short Scout
        if ctx.market_bias_4h == "bearish" and ctx.smc_trend_15m == "bear":
            if ctx.bos_detected_15m:
                return DecisionSummary(Action.OPEN_T1_SHORT, "BOS")
            if ctx.choch_detected_15m:
                return DecisionSummary(Action.OPEN_T1_SHORT, "CHoCH")

    # ── TRADE 2: SNIPER (Liquidity Sweep) ─────────────────────────────────
    # Phase 3/4: T2 recovery ONLY if T1 was stopped out (pnl < 0)
    if not ctx.t1_active and ctx.t1_stopped_out:
        # Gate 3: Structure Check
        # If trend flipped against us since T1 fail, abort T2
        if ctx.market_bias_4h == "bullish" and ctx.smc_trend_15m == "bear":
             return DecisionSummary(Action.HOLD, "REASON_STRUCTURE_BROKEN")
        if ctx.market_bias_4h == "bearish" and ctx.smc_trend_15m == "bull":
             return DecisionSummary(Action.HOLD, "REASON_STRUCTURE_BROKEN")

        # Sniper Entry Funnel
        if not ctx.liq_swept:
             return DecisionSummary(Action.HOLD, "REASON_NO_LIQUIDITY")
        
        if not ctx.fvg_side or ctx.fvg_filled:
             return DecisionSummary(Action.HOLD, "REASON_NO_FVG")

        # Phase 3: Scoring
        score = 0
        # 1. Session Score (+1 if London/NY)
        if ctx.session in ("london", "newyork"): score += 1
        # 2. RSI Score (+1 if deep retracement)
        if ctx.smc_trend_15m == "bull" and (ctx.rsi_14 or 50) < 40: score += 1
        if ctx.smc_trend_15m == "bear" and (ctx.rsi_14 or 50) > 60: score += 1
        # 3. FVG Age Score (+1 if fresh)
        if (ctx.fvg_age_bars or 0) < 10: score += 1
        # 4. Premium/Discount (+1)
        if ctx.smc_trend_15m == "bull" and (ctx.price_position or "").startswith("discount"): score += 1
        if ctx.smc_trend_15m == "bear" and (ctx.price_position or "").startswith("premium"): score += 1

        if score < MIN_SNIPER_SCORE:
            return DecisionSummary(Action.HOLD, "REASON_SCORE_TOO_LOW", {"score": score})

        # Bullish sweep (sweep of a low)
        if ctx.smc_trend_15m == "bull" and ctx.liq_side == "low":
            return DecisionSummary(Action.OPEN_T2_LONG, "LIQ_SWEEP", {"score": score})
        
        # Bearish sweep (sweep of a high)
        if ctx.smc_trend_15m == "bear" and ctx.liq_side == "high":
            return DecisionSummary(Action.OPEN_T2_SHORT, "LIQ_SWEEP", {"score": score})

    return DecisionSummary(Action.HOLD, "hold")


# ─────────────────────────────────────────────────────────────────────────────
# Row builder  (same fetching pattern as old strategy)
# ─────────────────────────────────────────────────────────────────────────────

def build_context_from_row(row: dict, t1_stopped_out: bool = False) -> ScoutSniperContext:
    """
    Converts a tick_features DB row (asyncpg Record or plain dict)
    into a ScoutSniperContext ready for make_decision.

    Accepts `t1_stopped_out` relay from the engine to track recovery status.
    """
    bid = float(row.get("bid", 0.0))
    ask = float(row.get("ask", 0.0))
    mid = (bid + ask) / 2.0

    return ScoutSniperContext(
        # ── Price ─────────────────────────────────────────────────────────
        mid=mid,
        bid=bid,
        ask=ask,
        spread=ask - bid,

        # ── Session / position ────────────────────────────────────────────
        session=row.get("session"),
        price_position=row.get("price_position"),
        bar_close=_safe_float(row.get("bar_close")),

        # ── FVG (1m Sniper entry) ─────────────────────────────────────────
        fvg_high=_safe_float(row.get("fvg_high")),
        fvg_low=_safe_float(row.get("fvg_low")),
        fvg_side=row.get("fvg_side"),
        fvg_filled=_safe_bool(row.get("fvg_filled")),
        fvg_age_bars=_safe_int(row.get("fvg_age_bars")),

        # ── Phase 1 SMC ───────────────────────────────────────────────────
        atr_20_1m=_safe_float(row.get("atr_20_1m")),
        atr_15_15m=_safe_float(row.get("atr_15_15m")),
        rsi_14=_safe_float(row.get("rsi_14")),
        prev_day_high=_safe_float(row.get("prev_day_high")),
        prev_day_low=_safe_float(row.get("prev_day_low")),
        current_session_high=_safe_float(row.get("current_session_high")),
        current_session_low=_safe_float(row.get("current_session_low")),
        prev_session_high=_safe_float(row.get("prev_session_high")),
        prev_session_low=_safe_float(row.get("prev_session_low")),
        session_boundary=_safe_bool(row.get("session_boundary")),
        n_confirmed_swing_highs_15m=_safe_int(row.get("n_confirmed_swing_highs_15m")) or 0,
        n_confirmed_swing_lows_15m=_safe_int(row.get("n_confirmed_swing_lows_15m")) or 0,

        # ── Phase 2 SMC ───────────────────────────────────────────────────
        smc_trend_15m=row.get("smc_trend_15m"),
        hh_15m=_safe_float(row.get("hh_15m")),
        ll_15m=_safe_float(row.get("ll_15m")),
        strong_low_15m=_safe_float(row.get("strong_low_15m")),
        strong_high_15m=_safe_float(row.get("strong_high_15m")),
        bos_detected_15m=_safe_bool(row.get("bos_detected_15m")),
        choch_detected_15m=_safe_bool(row.get("choch_detected_15m")),
        market_bias_4h=row.get("market_bias_4h"),
        fvg_timestamp=row.get("fvg_timestamp"),
        # ── Liquidity (Sniper entry) ──────────────────────────────────────
        liq_swept=_safe_bool(row.get("liq_swept")),
        liq_side=row.get("liq_side"),

        # ── Trade status ──────────────────────────────────────────────────
        t1_active=False,
        t2_active=False,
        t1_stopped_out=t1_stopped_out,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers  (identical pattern to old strategy + _safe_int added)
# ─────────────────────────────────────────────────────────────────────────────

def _any_none(*values) -> bool:
    return any(v is None for v in values)


def _derive_price_position(
    dist_high: Optional[float],
    dist_low: Optional[float],
) -> Optional[str]:
    """
    Infer premium/discount from distance ratio when not pre-tagged.
    dist_low / (dist_high + dist_low):
        < 0.25  → discount_extreme
        < 0.50  → discount
        < 0.75  → premium
        >= 0.75 → premium_extreme
    """
    if dist_high is None or dist_low is None:
        return None
    total = dist_high + dist_low
    if total == 0:
        return None
    ratio = dist_low / total
    if ratio < 0.25:
        return "discount_extreme"
    if ratio < 0.50:
        return "discount"
    if ratio < 0.75:
        return "premium"
    return "premium_extreme"


def _safe_float(val) -> Optional[float]:
    try:
        import math
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _safe_bool(val) -> bool:
    if val is None or pd.isna(val):
        return False
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