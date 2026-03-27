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

from dataclasses import dataclass
from typing import Optional

from src.backtest.backtest_engine import Action


# ─────────────────────────────────────────────────────────────────────────────
# Context dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScoutSniperContext:
    """
    Full tick context for the Scout & Sniper engine.
    All fields map 1-to-1 to tick_features DB columns.
    Existing columns are kept; new SMC columns are added below.
    """

    # ── Price ─────────────────────────────────────────────────────────────
    mid:    float
    bid:    float
    ask:    float
    spread: float

    # ── Indicators ────────────────────────────────────────────────────────
    rsi_14: Optional[float]
    atr_14: Optional[float]

    # ── Session & position (carried over from old strategy) ───────────────
    session:        Optional[str]   # killzone | london | new_york | asian
    price_position: Optional[str]   # premium_extreme | premium | discount | discount_extreme
    bar_close:      Optional[float]

    # ── Market structure (NEW — BOS / CHoCH) ─────────────────────────────
    structure_direction: Optional[str]   # "bullish" | "bearish"
    bos_detected:        Optional[bool]  # Break of Structure on this tick
    choch_detected:      Optional[bool]  # Change of Character on this tick

    # ── Fair Value Gap (NEW — FVG sniper entry) ───────────────────────────
    fvg_high:     Optional[float] = None  # top of nearest active FVG
    fvg_low:      Optional[float] = None  # bottom of nearest active FVG
    fvg_side:     Optional[str]    = None  # "bullish_fvg" | "bearish_fvg"
    fvg_filled:   bool = False             # True = already mitigated, skip
    fvg_age_bars: Optional[int]    = None  # bars since FVG formed

    # ── Phase 1 SMC (NEW — multi-timeframe & liquidity scoring) ───────────
    atr_20_1m:    Optional[float] = None  # FVG noise filter
    atr_15_15m:   Optional[float] = None  # structural volatility
    prev_day_high: Optional[float] = None
    prev_day_low:  Optional[float] = None
    current_session_high: Optional[float] = None
    current_session_low:  Optional[float] = None
    prev_session_high:    Optional[float] = None
    prev_session_low:     Optional[float] = None
    session_boundary:     bool = False
    n_confirmed_swing_highs_15m: int = 0
    n_confirmed_swing_lows_15m:  int = 0

    # ── Phase 2 SMC (New — structural nodes) ─────────────────────────────
    smc_trend_15m:   Optional[str]   = None
    hh_15m:          Optional[float] = None
    ll_15m:          Optional[float] = None
    strong_low_15m:  Optional[float] = None
    strong_high_15m: Optional[float] = None
    bos_detected_15m:   bool = False
    choch_detected_15m: bool = False
    market_bias_4h:    Optional[str] = None

    # ── Trade status (passed from engine) ────────────────────────────────
    t1_active:   bool = False
    t2_active:   bool = False

    # ── Liquidity (NEW — for sniper entry) ──────────────────────────────
    liq_swept:   bool = False
    liq_side:    Optional[str] = None # "high" | "low"


# ─────────────────────────────────────────────────────────────────────────────
# Thresholds  (tune during backtesting)
# ─────────────────────────────────────────────────────────────────────────────

MAX_SPREAD_ATR_RATIO = 0.25   # skip if spread > 25% of ATR
MAX_FVG_AGE_BARS     = 20     # discard stale FVGs older than N bars
LONG_RSI_MAX         = 55     # RSI ceiling for longs (not overbought)
SHORT_RSI_MIN        = 45     # RSI floor for shorts (not oversold)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def make_decision(ctx: ScoutSniperContext) -> Action:
    """
    Scout & Sniper execution engine.
    
    1. Scout (T1): Fires on 15m BOS/CHoCH. Must match 4H bias.
    2. Sniper (T2): Fires on 5m liquidity sweep IF T1 is already active.
    """

    # ── Gate 1: session ───────────────────────────────────────────────────
    if ctx.session not in ("killzone", "london"):
        pass

    # ── Gate 2: spread filter ─────────────────────────────────────────────
    if ctx.atr_14 and ctx.atr_14 > 0:
        if ctx.atr_14 is None or ctx.spread is None:
            return Action.HOLD
        if ctx.spread > ctx.atr_14 * MAX_SPREAD_ATR_RATIO:
            return Action.HOLD

    # ── Gate 3: 4H Bias ───────────────────────────────────────────────────
    if not ctx.market_bias_4h or ctx.market_bias_4h == "neutral":
        return Action.HOLD

    # ── TRADE 1: SCOUT (Structural Break) ─────────────────────────────────
    if not ctx.t1_active:
        # Long Scout
        if ctx.market_bias_4h == "bullish" and ctx.smc_trend_15m == "bull":
            if ctx.bos_detected_15m or ctx.choch_detected_15m:
                return Action.OPEN_T1_LONG
        
        # Short Scout
        if ctx.market_bias_4h == "bearish" and ctx.smc_trend_15m == "bear":
            if ctx.bos_detected_15m or ctx.choch_detected_15m:
                return Action.OPEN_T1_SHORT

    # ── TRADE 2: SNIPER (Liquidity Sweep) ─────────────────────────────────
    if ctx.t1_active and not ctx.t2_active:
        if ctx.liq_swept and ctx.liq_side is not None:
            # Bullish sweep (sweep of a low)
            if ctx.smc_trend_15m == "bull" and ctx.liq_side == "low":
                return Action.OPEN_T2_LONG
            
            # Bearish sweep (sweep of a high)
            if ctx.smc_trend_15m == "bear" and ctx.liq_side == "high":
                return Action.OPEN_T2_SHORT

    return Action.HOLD


# ─────────────────────────────────────────────────────────────────────────────
# Row builder  (same fetching pattern as old strategy)
# ─────────────────────────────────────────────────────────────────────────────

def build_context_from_row(row: dict) -> ScoutSniperContext:
    """
    Converts a tick_features DB row (asyncpg Record or plain dict)
    into a ScoutSniperContext ready for make_decision.

    Drop-in replacement for the old build_context_from_row — same
    call-site signature, same helper functions, extended fields only.
    """
    bid = float(row.get("bid", 0.0))
    ask = float(row.get("ask", 0.0))
    mid = (bid + ask) / 2.0

    dist_high = _safe_float(row.get("dist_to_nearest_high"))
    dist_low  = _safe_float(row.get("dist_to_nearest_low"))

    # Derive premium/discount if not pre-tagged by FeatureEngineer
    price_position = row.get("price_position") or _derive_price_position(
        dist_high, dist_low
    )

    return ScoutSniperContext(
        # ── Price ─────────────────────────────────────────────────────────
        mid=mid,
        bid=bid,
        ask=ask,
        spread=ask - bid,

        # ── Indicators ────────────────────────────────────────────────────
        rsi_14=_safe_float(row.get("rsi_14")),
        atr_14=_safe_float(row.get("atr_14")),

        # ── Session / position (unchanged columns) ────────────────────────
        session=row.get("session"),
        price_position=price_position,
        bar_close=_safe_float(row.get("bar_close")),

        # ── Market structure (new columns) ────────────────────────────────
        structure_direction=row.get("structure_direction"),
        bos_detected=_safe_bool(row.get("bos_detected")),
        choch_detected=_safe_bool(row.get("choch_detected")),

        # ── FVG (new columns) ─────────────────────────────────────────────
        fvg_high=_safe_float(row.get("fvg_high")),
        fvg_low=_safe_float(row.get("fvg_low")),
        fvg_side=row.get("fvg_side"),
        fvg_filled=bool(row.get("fvg_filled", False)),
        fvg_age_bars=_safe_int(row.get("fvg_age_bars")),

        # ── Phase 1 SMC (new columns) ─────────────────────────────────────
        atr_20_1m=_safe_float(row.get("atr_20_1m")),
        atr_15_15m=_safe_float(row.get("atr_15_15m")),
        prev_day_high=_safe_float(row.get("prev_day_high")),
        prev_day_low=_safe_float(row.get("prev_day_low")),
        current_session_high=_safe_float(row.get("current_session_high")),
        current_session_low=_safe_float(row.get("current_session_low")),
        prev_session_high=_safe_float(row.get("prev_session_high")),
        prev_session_low=_safe_float(row.get("prev_session_low")),
        session_boundary=bool(row.get("session_boundary", False)),
        n_confirmed_swing_highs_15m=_safe_int(row.get("n_confirmed_swing_highs_15m")) or 0,
        n_confirmed_swing_lows_15m=_safe_int(row.get("n_confirmed_swing_lows_15m")) or 0,

        # ── Phase 2 SMC (new columns) ─────────────────────────────────────
        smc_trend_15m=row.get("smc_trend_15m"),
        hh_15m=_safe_float(row.get("hh_15m")),
        ll_15m=_safe_float(row.get("ll_15m")),
        strong_low_15m=_safe_float(row.get("strong_low_15m")),
        strong_high_15m=_safe_float(row.get("strong_high_15m")),
        bos_detected_15m=bool(row.get("bos_detected_15m", False)),
        choch_detected_15m=bool(row.get("choch_detected_15m", False)),
        market_bias_4h=row.get("market_bias_4h"),

        # ── Liquidity (NEW for Sniper) ────────────────────────────────────
        liq_swept=bool(row.get("liq_swept", False)),
        liq_side=row.get("liq_side"),

        # Trade status should be injected by engine after building context
        t1_active=False,
        t2_active=False,
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


def _safe_bool(val) -> Optional[bool]:
    if val is None:
        return None
    if isinstance(val, bool):
        return val
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