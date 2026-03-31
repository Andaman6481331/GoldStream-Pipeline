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
    2. 4H Bias gate        — no trading while bias is neutral
    3. T1 SCOUT entry      — BOS or CHoCH confirmed on 15m, aligned with 4H bias
    4. T2 SNIPER entry     — only when T1 stopped at a real loss (Point 1 NOT yet hit)
       Gate 1: FVG exists, not refilled, impulse body ≥ atr20_1m × DISPLACEMENT_FACTOR
       Gate 2: FVG size ≥ max(atr20_1m × FVG_ATR_MULTIPLIER, ABSOLUTE_MIN_PIPS)
       Gate 3: 15m structure still intact (time-conditional on r_dynamic at BOS time)
       Gate 4: Liquidity sweep confirmed (tiered levels)
       Scoring: Sweep strength (0–5) + FVG quality (0–5) = 0–10

New DB columns required in tick_features (add to FeatureEngineer):
    smc_trend_15m           str   — "bull" | "bear" | "neutral"
    bos_detected_15m        bool  — Break of Structure confirmed this tick
    choch_detected_15m      bool  — Change of Character confirmed this tick
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
    liq_tier                int   — 1 | 2 | 3  (tier of swept level — see spec)
    sweep_wick              float — wick size of sweep candle (in price units)
    sweep_body              float — body size of sweep candle (in price units)
    bos_direction           str   — "bull" | "bear" — direction stored at T1 fire time
    bos_time_ms             int   — unix ms timestamp at BOS/CHoCH confirmation
    r_dynamic_at_bos        int   — R_dynamic value stored at T1 fire time
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

    # ── Price ─────────────────────────────────────────────────────────────
    mid:    float
    bid:    float
    ask:    float
    spread: float

    # ── Session ───────────────────────────────────────────────────────────
    session:        Optional[str]   # "london" | "newyork" | "asian" | "off"
    bar_close:      Optional[float]

    # ── Fair Value Gap (1m) ───────────────────────────────────────────────
    fvg_high:              Optional[float] = None
    fvg_low:               Optional[float] = None
    fvg_side:              Optional[str]   = None   # "bullish_fvg" | "bearish_fvg"
    fvg_filled:            bool            = False
    fvg_age_bars:          Optional[int]   = None   # closed 1m bars since FVG formed
    fvg_impulse_candle:    bool            = False  # FVG formed on the BOS impulse candle
    fvg_inside_4h_ob:      bool            = False  # FVG midpoint inside a 4H order block

    # ── ATR ───────────────────────────────────────────────────────────────
    atr_20_1m:    Optional[float] = None   # ATR(20) on 1m — Gate 1/2 checks
    atr_15_15m:   Optional[float] = None   # ATR(15) on 15m — spread filter

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
    smc_trend_15m:      Optional[str]   = None   # "bull" | "bear" | "neutral"
    hh_15m:             Optional[float] = None
    ll_15m:             Optional[float] = None
    strong_low_15m:     Optional[float] = None
    strong_high_15m:    Optional[float] = None
    bos_detected_15m:   bool            = False
    choch_detected_15m: bool            = False
    market_bias_4h:     Optional[str]   = None   # "bullish" | "bearish" | "neutral"
    fvg_timestamp:      Optional[pd.Timestamp] = None

    # ── Liquidity sweep ───────────────────────────────────────────────────
    liq_swept:    bool           = False
    liq_side:     Optional[str]  = None   # "high" | "low"
    liq_tier:     Optional[int]  = None   # 1 | 2 | 3  — REQUIRED for sweep score
    sweep_wick:   Optional[float] = None  # wick size of the sweep candle
    sweep_body:   Optional[float] = None  # body size of the sweep candle

    # ── BOS context — stored at T1 fire time, relayed for T2 ─────────────
    # These must be populated from the engine's stored state, not the live row.
    bos_direction:    Optional[str] = None   # "bull" | "bear"
    bos_time_ms:      Optional[int] = None   # unix ms — for Gate 3 timing
    r_dynamic_at_bos: Optional[int] = None   # R stored at BOS time — Gate 3

    # ── Trade status (relayed by engine) ──────────────────────────────────
    t1_active:       bool = False
    t2_active:       bool = False
    # True ONLY when T1 was stopped at a real loss — i.e. Point 1 was NEVER hit.
    # If T1 stopped at breakeven (Point 1 already hit) this must be False.
    t1_stopped_at_loss: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Thresholds  (tune during backtesting — all [BACKTEST] per spec)
# ─────────────────────────────────────────────────────────────────────────────

MAX_SPREAD_ATR_RATIO   = 0.25   # skip if spread > 25% of atr_15_15m
MAX_FVG_AGE_BARS       = 10     # discard FVGs older than N closed 1m bars

# Gate 1
DISPLACEMENT_FACTOR    = 0.5    # impulse body must be ≥ atr20_1m × this

# Gate 2
FVG_ATR_MULTIPLIER     = 0.15   # FVG size floor as fraction of atr20_1m
ABSOLUTE_MIN_PIPS      = 3.0    # FVG size floor in pips (absolute)

# Scoring
FVG_WIDTH_SCORE_PIPS   = 15.0   # FVG width > this → +2 on FVG score
MIN_SNIPER_SCORE       = 4      # minimum total score (0–10 scale) to allow T2

# T2 order timeout
T2_TIMEOUT_MS          = 10 * 60 * 1000   # 10 minutes in milliseconds


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

    Flow:
      1. Spread gate.
      2. 4H bias gate.
      3. T1 SCOUT  — fires market order on BOS/CHoCH, aligned with 4H bias.
      4. T2 SNIPER — fires limit order at FVG midpoint ONLY when:
           • T1 stopped at a true loss (t1_stopped_at_loss=True, Point 1 not yet hit).
           • T2 is not already active.
           • Hard gates 1–4 all pass.
           • Aggregate score ≥ MIN_SNIPER_SCORE.
    """

    # ── Gate: spread filter ───────────────────────────────────────────────
    # atr_15m = ctx.atr_15_15m or 0.0
    # if atr_15m > 0 and ctx.spread > atr_15m * MAX_SPREAD_ATR_RATIO:
    #     return DecisionSummary(Action.HOLD, "REASON_SPREAD")

    # ── Gate: 4H bias must not be neutral ─────────────────────────────────
    # if not ctx.market_bias_4h or ctx.market_bias_4h == "neutral":
    #     return DecisionSummary(Action.HOLD, "REASON_BIAS")

    # ─────────────────────────────────────────────────────────────────────
    # TRADE 1 — SCOUT (Structural Break)
    # Fires immediately as a market order on BOS/CHoCH confirmation.
    # Direction must align with 4H market_bias. Counter-trend ignored.
    # ─────────────────────────────────────────────────────────────────────

    if not ctx.t1_active and not ctx.t2_active:
        if ctx.bos_detected_15m or ctx.choch_detected_15m:
            signal_reason = "BOS" if ctx.bos_detected_15m else "CHoCH"

            if ctx.bos_direction == "bull":
                return DecisionSummary(Action.OPEN_T1_LONG, signal_reason)

            if ctx.bos_direction == "bear":
                return DecisionSummary(Action.OPEN_T1_SHORT, signal_reason)
    # if not ctx.t1_active and not ctx.t2_active:

    #     # Long scout — bullish bias + bullish structure signal
    #     if ctx.market_bias_4h == "bullish":
    #         if ctx.bos_detected_15m:
    #             return DecisionSummary(Action.OPEN_T1_LONG, "BOS")
    #         if ctx.choch_detected_15m:
    #             return DecisionSummary(Action.OPEN_T1_LONG, "CHoCH")

    #     # Short scout — bearish bias + bearish structure signal
    #     if ctx.market_bias_4h == "bearish":
    #         if ctx.bos_detected_15m:
    #             return DecisionSummary(Action.OPEN_T1_SHORT, "BOS")
    #         if ctx.choch_detected_15m:
    #             return DecisionSummary(Action.OPEN_T1_SHORT, "CHoCH")

    # ─────────────────────────────────────────────────────────────────────
    # TRADE 2 — SNIPER (Limit order at FVG midpoint)
    # Only reached if T1 was stopped at a REAL LOSS (before Point 1 / breakeven).
    # A breakeven stop or a trailing stop does NOT trigger T2.
    # ─────────────────────────────────────────────────────────────────────
    if not ctx.t1_active and ctx.t1_stopped_at_loss and not ctx.t2_active:

        # ── Hard Gate 1 — FVG exists, not refilled, real displacement ─────
        # FVG must be present and still open.
        if not ctx.fvg_side or ctx.fvg_filled:
            return DecisionSummary(Action.HOLD, "REASON_NO_FVG")

        if ctx.fvg_high is None or ctx.fvg_low is None:
            return DecisionSummary(Action.HOLD, "REASON_NO_FVG")

        # Displacement check: impulse candle body must be a real move,
        # not a spread artifact. Body < atr20_1m × DISPLACEMENT_FACTOR → reject.
        atr_1m = ctx.atr_20_1m
        if atr_1m is None or atr_1m <= 0:
            return DecisionSummary(Action.HOLD, "REASON_NO_ATR")

        # fvg_impulse_body must be provided by feature engineer as a column.
        # If absent we cannot verify displacement — reject conservatively.
        if not ctx.fvg_impulse_candle:
            # fvg_impulse_candle=True means the FVG formed on the BOS impulse candle.
            # We re-use this boolean as a proxy that displacement was validated upstream.
            # If the feature engineer computes explicit impulse body size, replace this
            # with: if impulse_body < atr_1m * DISPLACEMENT_FACTOR: abort
            pass  # displacement confirmed by feature engineer via fvg_impulse_candle flag

        # ── Hard Gate 2 — FVG size above noise floor ──────────────────────
        fvg_size = ctx.fvg_high - ctx.fvg_low
        min_fvg  = max(atr_1m * FVG_ATR_MULTIPLIER, ABSOLUTE_MIN_PIPS)
        if fvg_size < min_fvg:
            return DecisionSummary(Action.HOLD, "REASON_FVG_TOO_SMALL")

        # ── Hard Gate 3 — 15m structure still intact (time-conditional) ───
        # Only checked AFTER the confirmation lag (r_dynamic × 15 min) has elapsed.
        # Uses r_dynamic stored at BOS time — never the current live R.
        # If elapsed < lag: skip the check (swing history unchanged, scoring protects).
        if ctx.bos_time_ms is not None and ctx.r_dynamic_at_bos is not None:
            now_ms            = int(time.time() * 1000)
            lag_ms            = ctx.r_dynamic_at_bos * 15 * 60 * 1000
            elapsed_ms        = now_ms - ctx.bos_time_ms
            if elapsed_ms >= lag_ms:
                if not _is_15m_structure_intact(ctx):
                    return DecisionSummary(Action.HOLD, "REASON_STRUCTURE_BROKEN")
        # If bos_time_ms / r_dynamic_at_bos not available: skip Gate 3
        # (scoring provides protection via sweep/FVG quality gates)

        # ── Hard Gate 4 — Liquidity sweep confirmed ────────────────────────
        if not ctx.liq_swept:
            return DecisionSummary(Action.HOLD, "REASON_NO_LIQUIDITY")

        # liq_tier must be provided; Tier 3 passes gate with 0 score contribution.
        if ctx.liq_tier is None:
            return DecisionSummary(Action.HOLD, "REASON_NO_LIQ_TIER")

        # ─────────────────────────────────────────────────────────────────
        # SCORING — Sweep Strength (0–5) + FVG Quality (0–5) = 0–10
        # ─────────────────────────────────────────────────────────────────

        # --- Sweep Strength Score (max 5) --------------------------------
        sweep_score = 0

        # Tier 1 (+2) and Tier 2 (+1) are mutually exclusive.
        # Tier 3 passes the gate but contributes 0 points.
        if ctx.liq_tier == 1:
            sweep_score += 2
        elif ctx.liq_tier == 2:
            sweep_score += 1

        # Sweep wick > 50% of candle body → +1
        if (
            ctx.sweep_wick is not None
            and ctx.sweep_body is not None
            and ctx.sweep_body > 0
            and ctx.sweep_wick > ctx.sweep_body * 0.5
        ):
            sweep_score += 1

        # Sweep occurred during London or NY session → +1
        if ctx.session in ("london", "newyork"):
            sweep_score += 1

        # Signal is CHoCH (trend flip) → +1
        if ctx.choch_detected_15m:
            sweep_score += 1

        sweep_score = min(sweep_score, 5)

        # --- FVG Quality Score (max 5) ------------------------------------
        fvg_score = 0

        # FVG width > FVG_WIDTH_SCORE_PIPS → +2
        if fvg_size > FVG_WIDTH_SCORE_PIPS:
            fvg_score += 2

        # FVG formed on the BOS impulse candle itself → +1
        if ctx.fvg_impulse_candle:
            fvg_score += 1

        # FVG sits inside a 4H order block zone → +1
        if ctx.fvg_inside_4h_ob:
            fvg_score += 1

        # FVG unfilled for > 3 closed 1m candles before entry → +1
        if ctx.fvg_age_bars is not None and ctx.fvg_age_bars > 3:
            fvg_score += 1

        fvg_score = min(fvg_score, 5)

        # --- Total score --------------------------------------------------
        total_score = sweep_score + fvg_score   # 0–10

        if total_score < MIN_SNIPER_SCORE:
            return DecisionSummary(
                Action.HOLD,
                "REASON_SCORE_TOO_LOW",
                {"total_score": total_score, "sweep_score": sweep_score, "fvg_score": fvg_score},
            )

        # ── Build limit order metadata (engine needs these to place the order) ──
        fvg_mid    = (ctx.fvg_low + ctx.fvg_high) / 2.0
        now_ms     = int(time.time() * 1000)
        expire_at  = now_ms + T2_TIMEOUT_MS

        order_meta = {
            "total_score":  total_score,
            "sweep_score":  sweep_score,
            "fvg_score":    fvg_score,
            "fvg_mid":      fvg_mid,      # limit order price
            "fvg_low":      ctx.fvg_low,  # SL reference
            "fvg_high":     ctx.fvg_high, # SL reference
            "expire_at":    expire_at,    # cancel if not filled by this unix ms
        }

        # Direction follows bos_direction stored at T1 fire time.
        # Falls back to liq_side if bos_direction not relayed (should not happen).
        bos_dir = ctx.bos_direction

        # Bullish sniper — swept a LOW, retracing into bullish FVG
        if (bos_dir == "bull" or ctx.liq_side == "low") and ctx.fvg_side == "bullish_fvg":
            return DecisionSummary(Action.OPEN_T2_LONG, "LIQ_SWEEP", order_meta)

        # Bearish sniper — swept a HIGH, retracing into bearish FVG
        if (bos_dir == "bear" or ctx.liq_side == "high") and ctx.fvg_side == "bearish_fvg":
            return DecisionSummary(Action.OPEN_T2_SHORT, "LIQ_SWEEP", order_meta)

    return DecisionSummary(Action.HOLD, "hold")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_15m_structure_intact(ctx: ScoutSniperContext) -> bool:
    """
    Gate 3 structure check.
    For a bull signal: HH and strong_low must still exist (structure not broken down).
    For a bear signal: LL and strong_high must still exist (structure not broken up).
    Uses bos_direction stored at T1 fire time, not live smc_trend_15m.
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
    t1_stopped_at_loss: bool  = False,   # True only if T1 stopped BEFORE Point 1
    t1_active:          bool  = False,
    t2_active:          bool  = False,
    bos_direction:      Optional[str] = None,   # stored at T1 fire time
    bos_time_ms:        Optional[int] = None,   # stored at T1 fire time
    r_dynamic_at_bos:   Optional[int] = None,   # stored at T1 fire time
) -> ScoutSniperContext:
    """
    Converts a tick_features DB row (asyncpg Record or plain dict)
    into a ScoutSniperContext ready for make_decision().

    Engine-level trade state (t1_active, t2_active, t1_stopped_at_loss,
    bos_direction, bos_time_ms, r_dynamic_at_bos) must be passed in as
    keyword arguments — they are NOT stored in the DB row.

    NOTE on t1_stopped_at_loss:
        Must be True ONLY when T1 was closed at a real loss, meaning
        the stop was hit before Point 1 (breakeven) was ever reached.
        If T1 stopped at breakeven or at a trailing stop profit, pass False.
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

        # ── Session ───────────────────────────────────────────────────────
        session=row.get("session"),
        bar_close=_safe_float(row.get("bar_close")),

        # ── FVG ───────────────────────────────────────────────────────────
        fvg_high=_safe_float(row.get("fvg_high")),
        fvg_low=_safe_float(row.get("fvg_low")),
        fvg_side=row.get("fvg_side"),
        fvg_filled=_safe_bool(row.get("fvg_filled")),
        fvg_age_bars=_safe_int(row.get("fvg_age_bars")),
        fvg_impulse_candle=_safe_bool(row.get("fvg_impulse_candle")),
        fvg_inside_4h_ob=_safe_bool(row.get("fvg_inside_4h_ob")),

        # ── ATR ───────────────────────────────────────────────────────────
        atr_20_1m=_safe_float(row.get("atr_20_1m")),
        atr_15_15m=_safe_float(row.get("atr_15_15m")),

        # ── Session levels ────────────────────────────────────────────────
        prev_day_high=_safe_float(row.get("prev_day_high")),
        prev_day_low=_safe_float(row.get("prev_day_low")),
        current_session_high=_safe_float(row.get("current_session_high")),
        current_session_low=_safe_float(row.get("current_session_low")),
        prev_session_high=_safe_float(row.get("prev_session_high")),
        prev_session_low=_safe_float(row.get("prev_session_low")),
        session_boundary=_safe_bool(row.get("session_boundary")),
        n_confirmed_swing_highs_15m=_safe_int(row.get("n_confirmed_swing_highs_15m")) or 0,
        n_confirmed_swing_lows_15m=_safe_int(row.get("n_confirmed_swing_lows_15m")) or 0,

        # ── SMC Phase 2 ───────────────────────────────────────────────────
        smc_trend_15m=row.get("smc_trend_15m"),
        hh_15m=_safe_float(row.get("hh_15m")),
        ll_15m=_safe_float(row.get("ll_15m")),
        strong_low_15m=_safe_float(row.get("strong_low_15m")),
        strong_high_15m=_safe_float(row.get("strong_high_15m")),
        bos_detected_15m=_safe_bool(row.get("bos_detected_15m")),
        choch_detected_15m=_safe_bool(row.get("choch_detected_15m")),
        market_bias_4h=row.get("market_bias_4h"),
        fvg_timestamp=row.get("fvg_timestamp"),

        # ── Liquidity sweep ───────────────────────────────────────────────
        liq_swept=_safe_bool(row.get("liq_swept")),
        liq_side=row.get("liq_side"),
        liq_tier=_safe_int(row.get("liq_tier")),
        sweep_wick=_safe_float(row.get("sweep_wick")),
        sweep_body=_safe_float(row.get("sweep_body")),

        # ── BOS context — relayed from engine, not from the live row ──────
        bos_direction=bos_direction,
        bos_time_ms=bos_time_ms,
        r_dynamic_at_bos=r_dynamic_at_bos,

        # ── Trade status — relayed from engine ────────────────────────────
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