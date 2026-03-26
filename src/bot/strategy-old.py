"""
Strategy — SMC-aware decision engine
Replaces the naive RSI/EMA make_decision with a full SMC context-driven
approach that consumes all fields produced by FeatureEngineer v2.

Decision hierarchy:
    1. Session gate          — only trade killzone / london
    2. Liquidity gate        — only trade near confirmed, unswept, scored levels
    3. Spread gate           — skip thin-market ticks
    4. Premium/discount bias — long only in discount, short only in premium
    5. Distance ratio        — confirms which stop pool is closest
    6. RSI as confirmation   — momentum filter, not the primary signal
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Import Action from backtest engine — strategy is direction-agnostic
from src.backtest.backtest_engine import Action


# ── SMC Context ───────────────────────────────────────────────────────────────

@dataclass
class SMCContext:
    """
    Full context passed to make_decision.
    All fields map 1-to-1 to columns in tick_features (Gold layer output).
    """
    # ── Price ─────────────────────────────────────────────────────────────
    mid:    float
    bid:    float
    ask:    float
    spread: float

    # ── Indicators ────────────────────────────────────────────────────────
    rsi_14: Optional[float]
    atr_14: Optional[float]

    # ── Liquidity (FeatureEngineer v2) ────────────────────────────────────
    liq_level:             Optional[float]
    liq_type:              Optional[str]    # swing_high | swing_low | round_number
    liq_side:              Optional[str]    # buystops_above | sellstops_below
    liq_score:             Optional[float]  # confluence score (higher = stronger)
    liq_confirmed:         Optional[bool]   # structure break confirmed
    liq_swept:             Optional[bool]   # already consumed — skip
    dist_to_nearest_high:  Optional[float]  # price distance to nearest level above
    dist_to_nearest_low:   Optional[float]  # price distance to nearest level below

    # ── Market context ────────────────────────────────────────────────────
    session:        Optional[str]   # killzone | london | new_york | asian
    price_position: Optional[str]   # premium_extreme | premium | discount | discount_extreme
    bar_close:      Optional[float]


# ── Decision thresholds (tune these during backtesting) ───────────────────────

MIN_LIQ_SCORE         = 4      # minimum confluence score to consider a level
MAX_SPREAD_ATR_RATIO  = 0.30   # skip if spread > 30% of ATR (thin market)
LONG_RSI_MAX          = 45     # RSI ceiling for long entries (momentum weakening)
SHORT_RSI_MIN         = 55     # RSI floor for short entries (momentum strengthening)
DIST_LONG_MAX_ATR     = 0.50   # long only if within 50% of ATR from sell-stop level
DIST_SHORT_MAX_ATR    = 0.50   # short only if within 50% of ATR from buy-stop level
LIQ_PRESSURE_LONG_MAX = 0.30   # price must be very close to sell-stop zone
LIQ_PRESSURE_SHORT_MIN= 0.70   # price must be very close to buy-stop zone


# ── Main entry point ──────────────────────────────────────────────────────────

def make_decision(ctx: SMCContext) -> Action:
    """
    SMC-aware strategy. Returns Action enum consumed by BacktestEngine.

    Long thesis:
        Price is in a discount zone, approaching a confirmed sell-stop level
        (swing low), session is active, RSI confirms weakening momentum.
        Smart money likely to sweep the sell stops then reverse up.

    Short thesis:
        Price is in a premium zone, approaching a confirmed buy-stop level
        (swing high), session is active, RSI confirms elevated momentum.
        Smart money likely to sweep the buy stops then reverse down.
    """

    # ── Gate 1: session ───────────────────────────────────────────────────
    if ctx.session not in ("killzone", "london"):
        return Action.HOLD

    # ── Gate 2: liquidity context must be complete ────────────────────────
    if _any_none(
        ctx.liq_level, ctx.liq_score, ctx.liq_confirmed,
        ctx.liq_swept, ctx.liq_side,
        ctx.dist_to_nearest_high, ctx.dist_to_nearest_low,
    ):
        return Action.HOLD

    # ── Gate 3: level must be confirmed, unswept, and strong enough ───────
    if ctx.liq_swept:
        return Action.HOLD
    if not ctx.liq_confirmed:
        return Action.HOLD
    if ctx.liq_score < MIN_LIQ_SCORE:
        return Action.HOLD

    # ── Gate 4: spread filter ─────────────────────────────────────────────
    if ctx.atr_14 and ctx.atr_14 > 0:
        if ctx.spread > ctx.atr_14 * MAX_SPREAD_ATR_RATIO:
            return Action.HOLD

    # ── Gate 5: price position must be known ─────────────────────────────
    if ctx.price_position is None:
        return Action.HOLD

    # ── Directional pressure ratio ────────────────────────────────────────
    total_dist = (ctx.dist_to_nearest_high or 0) + (ctx.dist_to_nearest_low or 0)
    if total_dist == 0:
        return Action.HOLD

    # 0 = right at sell-stop zone (sweep down imminent)
    # 1 = right at buy-stop zone  (sweep up imminent)
    liq_pressure = (ctx.dist_to_nearest_low or 0) / total_dist

    atr = ctx.atr_14 or 1.0

    # ── LONG: discount zone + near sell-stops + RSI weakening ────────────
    long_ok = (
        ctx.price_position in ("discount", "discount_extreme")
        and ctx.liq_side == "sellstops_below"
        and liq_pressure < LIQ_PRESSURE_LONG_MAX
        and ctx.rsi_14 is not None
        and ctx.rsi_14 < LONG_RSI_MAX
        and (ctx.dist_to_nearest_low or 999) < atr * DIST_LONG_MAX_ATR
    )

    # ── SHORT: premium zone + near buy-stops + RSI elevated ──────────────
    short_ok = (
        ctx.price_position in ("premium", "premium_extreme")
        and ctx.liq_side == "buystops_above"
        and liq_pressure > LIQ_PRESSURE_SHORT_MIN
        and ctx.rsi_14 is not None
        and ctx.rsi_14 > SHORT_RSI_MIN
        and (ctx.dist_to_nearest_high or 999) < atr * DIST_SHORT_MAX_ATR
    )

    if long_ok:
        return Action.OPEN_LONG
    if short_ok:
        return Action.OPEN_SHORT
    return Action.HOLD


# ── Helpers ───────────────────────────────────────────────────────────────────

def _any_none(*values) -> bool:
    return any(v is None for v in values)


def build_context_from_row(row: dict) -> SMCContext:
    """
    Convenience builder: converts a tick_features DB row (asyncpg Record
    or plain dict) into an SMCContext ready for make_decision.

    Also derives price_position from dist columns if not pre-computed.
    """
    bid = float(row.get("bid", 0.0))
    ask = float(row.get("ask", 0.0))
    mid = (bid + ask) / 2.0

    dist_high = _safe_float(row.get("dist_to_nearest_high"))
    dist_low  = _safe_float(row.get("dist_to_nearest_low"))

    # Derive premium/discount if not already in row
    price_position = row.get("price_position") or _derive_price_position(
        dist_high, dist_low
    )

    return SMCContext(
        mid=mid,
        bid=bid,
        ask=ask,
        spread=ask - bid,
        rsi_14=_safe_float(row.get("rsi_14")),
        atr_14=_safe_float(row.get("atr_14")),
        liq_level=_safe_float(row.get("liq_level")),
        liq_type=row.get("liq_type"),
        liq_side=row.get("liq_side"),
        liq_score=_safe_float(row.get("liq_score")),
        liq_confirmed=_safe_bool(row.get("liq_confirmed")),
        liq_swept=_safe_bool(row.get("liq_swept")),
        dist_to_nearest_high=dist_high,
        dist_to_nearest_low=dist_low,
        session=row.get("session"),
        price_position=price_position,
        bar_close=_safe_float(row.get("bar_close")),
    )


def _derive_price_position(
    dist_high: Optional[float],
    dist_low: Optional[float],
) -> Optional[str]:
    """
    Infer premium/discount from distance ratio when not pre-tagged.
    dist_low / (dist_high + dist_low) < 0.25 → discount_extreme
                                      < 0.50 → discount
                                      < 0.75 → premium
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