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
    fvg_high:     Optional[float]  # top of nearest active FVG
    fvg_low:      Optional[float]  # bottom of nearest active FVG
    fvg_side:     Optional[str]    # "bullish_fvg" | "bearish_fvg"
    fvg_filled:   Optional[bool]   # True = already mitigated, skip
    fvg_age_bars: Optional[int]    # bars since FVG formed


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

    ── LONG thesis ──────────────────────────────────────────────────────────
        Scout:  Bullish CHoCH or BOS detected → structure shifted upward.
        Sniper: Price retraces into a fresh bullish FVG (imbalance below
                current price formed during the impulse move up).
                Enter long from FVG low/mid — stop below FVG low.

    ── SHORT thesis ─────────────────────────────────────────────────────────
        Scout:  Bearish CHoCH or BOS detected → structure shifted downward.
        Sniper: Price retraces into a fresh bearish FVG (imbalance above
                current price formed during the impulse move down).
                Enter short from FVG high/mid — stop above FVG high.
    """

    # ── Gate 1: session ───────────────────────────────────────────────────
    if ctx.session not in ("killzone", "london"):
        return Action.HOLD

    # ── Gate 2: spread filter ─────────────────────────────────────────────
    if ctx.atr_14 and ctx.atr_14 > 0:
        if ctx.spread > ctx.atr_14 * MAX_SPREAD_ATR_RATIO:
            return Action.HOLD

    # ── Gate 3: structure bias must be established ────────────────────────
    if ctx.structure_direction is None:
        return Action.HOLD

    # Optional: You could add a check for a "recent" break here if desired,
    # but structure_direction forward-fills the latest bias.

    # ── Gate 4: FVG must be valid and fresh ───────────────────────────────
    if _any_none(ctx.fvg_high, ctx.fvg_low, ctx.fvg_side, ctx.fvg_filled):
        return Action.HOLD

    if ctx.fvg_filled:
        return Action.HOLD

    if ctx.fvg_age_bars is not None and ctx.fvg_age_bars > MAX_FVG_AGE_BARS:
        return Action.HOLD

    # ── Gate 5: price must be INSIDE the FVG (sniper entry) ───────────────
    price_in_fvg = ctx.fvg_low <= ctx.mid <= ctx.fvg_high
    if not price_in_fvg:
        return Action.HOLD

    # ── Gate 6: price position must be known ─────────────────────────────
    if ctx.price_position is None:
        return Action.HOLD

    # ── LONG: bullish structure + bullish FVG + discount zone ────────────
    long_ok = (
        ctx.structure_direction == "bullish"
        and ctx.fvg_side == "bullish_fvg"
        and ctx.price_position in ("discount", "discount_extreme")
        and (ctx.rsi_14 is None or ctx.rsi_14 < LONG_RSI_MAX)
    )

    # ── SHORT: bearish structure + bearish FVG + premium zone ────────────
    short_ok = (
        ctx.structure_direction == "bearish"
        and ctx.fvg_side == "bearish_fvg"
        and ctx.price_position in ("premium", "premium_extreme")
        and (ctx.rsi_14 is None or ctx.rsi_14 > SHORT_RSI_MIN)
    )

    if long_ok:
        return Action.OPEN_LONG
    if short_ok:
        return Action.OPEN_SHORT
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
        fvg_filled=_safe_bool(row.get("fvg_filled")) or False,
        fvg_age_bars=_safe_int(row.get("fvg_age_bars")),
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
        return int(val)
    except (TypeError, ValueError):
        return None