"""
Backtesting Engine — Event-Based  (v3)
Iterates through enriched tick_features and drives the Scout & Sniper strategy.

Architecture:
  - T1 (Scout)  : market order on BOS/CHoCH confirmation.
  - T2 (Sniper) : PENDING limit order at FVG midpoint. Filled only when price
                  reaches fvg_mid. Cancelled on FVG refill, timeout, or bias flip.
  - Trailing SL : fractal-based on confirmed 1m swing points (L=R=3), one-way only.
                  Activated ONLY after Point 2 is hit.
  - Point 1     : moves SL to breakeven. Until Point 1 hit, original SL is live.
  - Point 2     : activates the fractal trailing SL mechanic.
  - SL (T1)     : sweep zone candle extreme ± T1_BUFFER_PIPS.
  - SL (T2)     : fvg_low − T2_BUFFER_PIPS (long) / fvg_high + T2_BUFFER_PIPS (short).
  - T2 TP       : optional structural TP at nearest Tier 1/2 level if R:R ≥ MIN_TP_RR.
  - 1% rule     : T1 + T2 combined max loss ≤ 1% of account. Lot sized accordingly.

Fixes vs v2:
  - T2 is now a pending limit order, not a market fill (critical)
  - Trailing SL is fractal swing-point based, not fixed-distance ratchet (critical)
  - Point 1 → BE move and Point 2 → trail activation are actually enforced (critical)
  - T1 SL uses sweep_candle extreme + T1_BUFFER_PIPS (critical)
  - T2 SL uses fvg_low/high + T2_BUFFER_PIPS (critical)
  - T2 structural TP implemented (critical)
  - 1% combined max loss rule enforced via lot sizing (critical)
  - _t1_stopped_out resets on T2 resolution only, not on bar boundary (high)
  - BOS dedup fixed — only suppresses after a trade was actually taken (high)
  - T2 fill uses fvg_mid from metadata, not live ask/bid (high)
  - FVG-refill cancellation during T2 pending state (high)
  - equity_curve updated on trailing stop closes (high)
  - max_drawdown and sharpe_ratio calculated from equity curve (medium)
  - PnL reported in both raw price units and dollar terms via lot size (medium)
  - atr_14 fallback removed — atr_15_15m only, warn if missing (medium)
  - List imported from typing (medium)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.gold.duckdb_store import DuckDBStore

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy parameters  (all [BACKTEST] per spec — tune via backtesting)
# ─────────────────────────────────────────────────────────────────────────────

T1_BUFFER_PIPS   = 4.0    # SL buffer beyond sweep zone extreme
T2_BUFFER_PIPS   = 2.5    # SL buffer beyond FVG boundary
P1_ATR_FACTOR    = 0.3    # Point 1 = entry ± atr15_15m × this
P2_ATR_FACTOR    = 1.25   # Point 2 = entry ± atr15_15m × this
P2_P1_MIN_GAP    = 0.3    # Point 2 must be ≥ Point1 ± atr15_15m × this from Point 1
TRAIL_BUFFER_PIPS = 2.5   # buffer below/above confirmed 1m swing point for trail SL
MIN_TP_RR        = 2.0    # T2 structural TP only if R:R ≥ this
T2_TIMEOUT_MS    = 10 * 60 * 1000   # 10 minutes in milliseconds
MAX_ACCOUNT_RISK = 0.01   # 1% of account — combined T1+T2 max loss
MIN_SNIPER_SCORE = 4      # with-trend T2 minimum score
MIN_SNIPER_SCORE_COUNTER = 7 # counter-trend T2 minimum score


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class Position(Enum):
    FLAT  = "FLAT"
    LONG  = "LONG"
    SHORT = "SHORT"

class Action(Enum):
    HOLD          = "HOLD"
    OPEN_T1_LONG  = "OPEN_T1_LONG"
    OPEN_T1_SHORT = "OPEN_T1_SHORT"
    OPEN_T2_LONG  = "OPEN_T2_LONG"
    OPEN_T2_SHORT = "OPEN_T2_SHORT"
    CLOSE_T1      = "CLOSE_T1"
    CLOSE_T2      = "CLOSE_T2"
    CLOSE_ALL     = "CLOSE_ALL"


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TickEvent:
    """
    Full snapshot of a single tick passed to the strategy callback.
    All fields map 1-to-1 to tick_features columns.
    """
    timestamp_utc:  datetime
    symbol:         str
    bid:            float
    ask:            float
    spread:         float
    mid:            float
    volume:         float
    volume_usd:     Optional[float]

    rsi_14:         Optional[float]

    bar_open:       Optional[float]
    bar_high:       Optional[float]
    bar_low:        Optional[float]
    bar_close:      Optional[float]

    liq_level:              Optional[float]
    liq_type:               Optional[str]
    liq_side:               Optional[str]
    liq_score:              Optional[float]
    liq_confirmed:          Optional[bool]
    liq_swept:              Optional[bool]
    dist_to_nearest_high:   Optional[float]
    dist_to_nearest_low:    Optional[float]

    session:        Optional[str]
    price_position: Optional[str]

    current_position:   Position        = Position.FLAT
    entry_price:        Optional[float] = None
    trailing_stop:      Optional[float] = None
    unrealised_pnl:     float           = 0.0


@dataclass
class TradeState:
    """Mutable state for a single active (or pending) trade."""
    type:           str             = "T1"       # "T1" | "T2"
    trade_pair_id:  int             = 0
    position:       Position        = Position.FLAT
    entry_price:    Optional[float] = None
    sl:             Optional[float] = None       # live stop-loss price
    tp:             Optional[float] = None       # structural TP (T2 only; None = trail only)
    entry_time:     Optional[datetime] = None
    entry_spread:   Optional[float] = None
    peak_price:     Optional[float] = None

    # Entry candle extremes — for Point 1 calculation
    entry_candle_high: float = 0.0
    entry_candle_low:  float = 0.0

    # Point 1 / Point 2 prices and status
    point1_price:   float = 0.0
    point2_price:   float = 0.0
    point1_hit:     bool  = False   # True → SL moved to breakeven
    point2_hit:     bool  = False   # True → fractal trailing SL activated

    # Trailing SL state — only active after point2_hit
    trailing_active: bool  = False

    # T2 pending limit order fields
    pending:        bool            = False
    fvg_mid:        Optional[float] = None   # limit order price
    fvg_low:        Optional[float] = None   # SL reference for T2 long
    fvg_high:       Optional[float] = None   # SL reference for T2 short
    expire_at_ms:   Optional[int]   = None   # unix ms — cancel if not filled by this

    # Sizing
    lot:            float = 0.01             # computed via 1% rule at entry
    risk_distance:  float = 0.0             # abs(entry - sl), set once at entry

    entry_reason:   str = ""
    session:        str = ""
    sl_pips:        float = 0.0
    bos_direction:  str = ""                # "bull" | "bear" — stored at BOS time


@dataclass
class CompletedTrade:
    """Record of a completed round-trip trade."""
    symbol:           str
    type:             str        # "T1" | "T2"
    trade_pair_id:    int
    direction:        Position
    entry_time:       datetime
    exit_time:        datetime
    entry_price:      float
    exit_price:       float
    pnl:              float      # raw price-unit PnL × lot (dollar-equivalent)
    pnl_pips:         float
    pnl_raw:          float      # exit - entry in raw price units (for reference)
    lot:              float
    exit_reason:      str        # "sl" | "tp" | "trailing_stop" | "strategy" | "wide_spread" | "end_of_data"
    entry_reason:     str        # "BOS" | "CHoCH" | "LIQ_SWEEP"
    session:          str
    point1_hit:       bool
    point2_hit:       bool
    hold_time_m:      float
    sl_pips_at_entry: float


@dataclass
class BacktestResult:
    """Summary of a completed backtest run."""
    total_ticks:        int
    total_trades:       int
    winning_trades:     int
    losing_trades:      int
    gross_pnl:          float
    win_rate:           float
    avg_pnl_per_trade:  float
    max_drawdown:       float
    profit_factor:      float
    sharpe_ratio:       float
    trades:             List[CompletedTrade] = field(default_factory=list)

    t1_wins:        int  = 0
    t1_losses:      int  = 0
    t2_wins:        int  = 0
    t2_losses:      int  = 0
    gate_funnel:    dict = field(default_factory=dict)
    session_stats:  Dict[str, dict] = field(default_factory=dict)
    
    # Advanced reporting
    bos_choch_events: List[Dict[str, Any]] = field(default_factory=list)
    thresholds:       Dict[str, Any]       = field(default_factory=dict)

    def __str__(self) -> str:
        def _wr(w, t): return f"{w/t:.1%}" if t > 0 else "n/a"
        lines = [
            "=" * 60,
            "  BACKTEST SUMMARY",
            "=" * 60,
            f"  Trades           : {self.total_trades} (Win Rate: {_wr(self.winning_trades, self.total_trades)})",
            f"  Gross PnL        : {self.gross_pnl:+.2f}",
            f"  Profit Factor    : {self.profit_factor:.2f}",
            f"  Sharpe Ratio     : {self.sharpe_ratio:.3f}",
            f"  Max Drawdown     : {self.max_drawdown:.2f}",
            "",
            "  STRATEGY CATEGORIES",
            "-" * 30,
            f"  T1 (Scout)  : {self.t1_wins + self.t1_losses:3d} trd | WR: {_wr(self.t1_wins, self.t1_wins + self.t1_losses)}",
            f"  T2 (Sniper) : {self.t2_wins + self.t2_losses:3d} trd | WR: {_wr(self.t2_wins, self.t2_wins + self.t2_losses)}",
            "",
            "  T2 SNIPER FUNNEL (Rejections)",
            "-" * 30,
        ]
        for reason, count in self.gate_funnel.items():
            lines.append(f"  {reason:30s}: {count}")
        lines += [
            "",
            "  SESSION PERFORMANCE",
            "-" * 30,
        ]
        for sess, stats in self.session_stats.items():
            lines.append(
                f"  {sess:10s}: {stats['total']:3d} trd | WR: {_wr(stats['wins'], stats['total'])}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Event-based backtesting engine for the Scout & Sniper strategy.

    Trade mechanics:
      T1 — market order at BOS/CHoCH confirmation tick.
      T2 — pending limit order at fvg_mid; filled only when price touches the
           limit level. Cancelled on FVG refill, T2_TIMEOUT, or bias flip.
      SL — structural for both trades (not ATR-generic).
      Trailing — fractal 1m swing-point based, activated only after Point 2.
      Lot sizing — 1% account risk rule, T1 + T2 combined.

    Usage:
        engine = BacktestEngine(store=store, symbol="XAUUSD")
        result = engine.run(from_dt=..., to_dt=...)
        print(result)
    """

    PIP_SIZE         = 0.10    # XAUUSD: 1 pip = $0.10
    LOT_PIP_VALUE    = 10.0    # $10 per pip per standard lot (100 oz)
    WIDE_SPREAD_PIPS = 20.0    # force-close threshold

    def __init__(
        self,
        store:           "DuckDBStore",
        symbol:          str   = "XAUUSD",
        initial_capital: float = 10_000.0,
    ):
        self.store           = store
        self.symbol          = symbol
        self.initial_capital = initial_capital

        # BOS deduplication: set ONLY after a trade was actually opened.
        # Never reset on a bare bar boundary — only reset when a new T1 opens.
        self._bos_taken_bar: Optional[datetime] = None

        # Recovery relay — True from T1 real loss until T2 resolves.
        # Reset on: T2 fill, T2 cancel (timeout/refill/bias), or new T1 open.
        self._t1_stopped_at_loss: bool = False

        # BOS context stored at T1 fire time — relayed to strategy for Gate 3.
        self._bos_direction:    Optional[str] = None
        self._bos_time_ms:      Optional[int] = None
        self._r_dynamic_at_bos: Optional[int] = None

    # ── Data loading ──────────────────────────────────────────────────────────

    def load_ticks(self, from_dt: datetime, to_dt: datetime) -> pd.DataFrame:
        df = self.store.query_features(self.symbol, from_dt, to_dt)
        if df.empty:
            logger.warning(
                f"[BacktestEngine] No data for {self.symbol} "
                f"between {from_dt} and {to_dt}"
            )
        return df

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(
        self,
        from_dt:  Optional[datetime]      = None,
        to_dt:    Optional[datetime]      = None,
        ticks_df: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """
        Main backtest loop. Calls make_decision each tick and manages the full
        trade lifecycle including T2 pending-order state.
        """
        if ticks_df is None:
            if from_dt is None or to_dt is None:
                raise ValueError("Provide either ticks_df or both from_dt and to_dt")
            ticks_df = self.load_ticks(from_dt, to_dt)

        if ticks_df.empty:
            logger.error("[BacktestEngine] No ticks — aborting")
            return BacktestResult(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        from src.bot.strategy_scout_sniper import build_context_from_row, make_decision

        active_trades: Dict[str, TradeState] = {}   # "T1" | "T2"
        trades:        List[CompletedTrade]  = []
        equity_curve:  List[float]           = [self.initial_capital]
        gate_funnel:   Dict[str, int]        = {}
        _pair_id = 1

        logger.info(f"[BacktestEngine] Starting run over {len(ticks_df):,} ticks")

        bos_choch_list: List[dict] = []
        # Capture current thresholds for reporting
        thresholds = {
            "T1_BUFFER_PIPS":    T1_BUFFER_PIPS,
            "T2_BUFFER_PIPS":    T2_BUFFER_PIPS,
            "P1_ATR_FACTOR":     P1_ATR_FACTOR,
            "P2_ATR_FACTOR":     P2_ATR_FACTOR,
            "MAX_ACCOUNT_RISK":  MAX_ACCOUNT_RISK,
            "MIN_SNIPER_SCORE":  MIN_SNIPER_SCORE,
            "MIN_SNIPER_SCORE_COUNTER": MIN_SNIPER_SCORE_COUNTER,
            "T2_TIMEOUT_MS":     T2_TIMEOUT_MS,
        }

        for _, row in ticks_df.iterrows():
            row_dict = dict(row)
            bid      = _safe_float(row_dict.get("bid"))  or 0.0
            ask      = _safe_float(row_dict.get("ask"))  or 0.0
            mid      = (bid + ask) / 2.0
            spread   = ask - bid
            ts       = pd.Timestamp(row_dict["timestamp_utc"]).to_pydatetime()
            ts_ms    = int(ts.timestamp() * 1000)

            # Current account equity for lot sizing
            current_equity = equity_curve[-1]

            # ── Wide spread — force-close all active trades ───────────────
            if self._is_wide_spread(spread):
                for tid in list(active_trades.keys()):
                    tstate = active_trades.pop(tid)
                    if tstate.pending:
                        continue    # pending order — just drop it
                    trade = self._close_trade(tstate, bid, ask, ts, "wide_spread")
                    trades.append(trade)
                    equity_curve.append(equity_curve[-1] + trade.pnl)
                    if tstate.type == "T1":
                        self._t1_stopped_at_loss = trade.pnl < 0
                continue

            # ── T2 pending order management ───────────────────────────────
            if "T2" in active_trades and active_trades["T2"].pending:
                t2 = active_trades["T2"]
                cancelled = False

                # 1. Timeout
                if ts_ms >= t2.expire_at_ms:
                    logger.debug(f"[BacktestEngine] T2 cancelled: timeout")
                    active_trades.pop("T2")
                    self._t1_stopped_at_loss = False
                    cancelled = True

                # 2. Bias flip
                if not cancelled:
                    live_bias = row_dict.get("market_bias_4h")
                    if live_bias and live_bias != "neutral":
                        bias_flipped = (
                            (t2.bos_direction == "bull" and live_bias == "bearish") or
                            (t2.bos_direction == "bear" and live_bias == "bullish")
                        )
                        if bias_flipped:
                            logger.debug(f"[BacktestEngine] T2 cancelled: bias flip")
                            active_trades.pop("T2")
                            self._t1_stopped_at_loss = False
                            cancelled = True

                # 3. FVG refilled — price closed back through the gap
                if not cancelled:
                    bar_close = _safe_float(row_dict.get("bar_close"))
                    if bar_close is not None:
                        fvg_refilled = _safe_bool(row_dict.get("fvg_filled"))
                        if fvg_refilled:
                            logger.debug(f"[BacktestEngine] T2 cancelled: FVG refilled")
                            active_trades.pop("T2")
                            self._t1_stopped_at_loss = False
                            cancelled = True

                # 4. Price reached limit — fill at fvg_mid
                if not cancelled and t2.fvg_mid is not None:
                    filled = (
                        (t2.position == Position.LONG  and ask <= t2.fvg_mid) or
                        (t2.position == Position.SHORT and bid >= t2.fvg_mid)
                    )
                    if filled:
                        fill_price = t2.fvg_mid
                        self._activate_pending_t2(
                            t2, fill_price, ts, spread,
                            row_dict, current_equity,
                        )
                        logger.debug(
                            f"[BacktestEngine] T2 FILLED at {fill_price:.5f}"
                        )
                        self._t1_stopped_at_loss = False

            # ── Active trade lifecycle ────────────────────────────────────
            to_remove: List[str] = []
            for tid, tstate in active_trades.items():
                if tstate.pending:
                    continue    # handled above

                # SL hit
                if tstate.sl is not None:
                    sl_hit = (
                        (tstate.position == Position.LONG  and bid <= tstate.sl) or
                        (tstate.position == Position.SHORT and ask >= tstate.sl)
                    )
                    if sl_hit:
                        trade = self._close_trade(tstate, bid, ask, ts, "sl")
                        trades.append(trade)
                        equity_curve.append(equity_curve[-1] + trade.pnl)
                        if tstate.type == "T1":
                            # T2 recovery only if T1 stopped BEFORE Point 1 (real loss)
                            self._t1_stopped_at_loss = (not tstate.point1_hit) and (trade.pnl < 0)
                        to_remove.append(tid)
                        continue

                # TP hit (T2 structural TP only — T1 has no TP)
                if tstate.tp is not None:
                    tp_hit = (
                        (tstate.position == Position.LONG  and bid >= tstate.tp) or
                        (tstate.position == Position.SHORT and ask <= tstate.tp)
                    )
                    if tp_hit:
                        trade = self._close_trade(tstate, bid, ask, ts, "tp")
                        trades.append(trade)
                        equity_curve.append(equity_curve[-1] + trade.pnl)
                        to_remove.append(tid)
                        continue

                # Point 1 — move SL to breakeven (once only)
                if not tstate.point1_hit and tstate.point1_price is not None:
                    p1_hit = (
                        (tstate.position == Position.LONG  and bid >= tstate.point1_price) or
                        (tstate.position == Position.SHORT and ask <= tstate.point1_price)
                    )
                    if p1_hit:
                        tstate.point1_hit = True
                        if tstate.entry_price is not None:
                            tstate.sl = tstate.entry_price   # move to breakeven
                        logger.debug(
                            f"[BacktestEngine] {tid} Point 1 hit — SL moved to BE @ {tstate.sl:.5f}"
                        )

                # Point 2 — activate fractal trailing SL
                if tstate.point1_hit and not tstate.point2_hit and tstate.point2_price is not None:
                    p2_hit = (
                        (tstate.position == Position.LONG  and bid >= tstate.point2_price) or
                        (tstate.position == Position.SHORT and ask <= tstate.point2_price)
                    )
                    if p2_hit:
                        tstate.point2_hit     = True
                        tstate.trailing_active = True
                        logger.debug(
                            f"[BacktestEngine] {tid} Point 2 hit — fractal trail activated"
                        )

                # Fractal trailing SL update — only after Point 2 activated
                if tstate.trailing_active:
                    self._update_fractal_trail(tstate, row_dict)

            for tid in to_remove:
                active_trades.pop(tid, None)

            # ── Strategy decision ─────────────────────────────────────────
            current_15m_bar = ts.replace(
                minute=(ts.minute // 15) * 15, second=0, microsecond=0
            )

            ctx = build_context_from_row(
                row_dict,
                t1_stopped_at_loss=self._t1_stopped_at_loss,
                t1_active=("T1" in active_trades),
                t2_active=("T2" in active_trades),
                bos_direction=self._bos_direction,
                bos_time_ms=self._bos_time_ms,
                r_dynamic_at_bos=self._r_dynamic_at_bos,
            )

            # BOS/CHoCH dedup — suppress signals only if we ALREADY took a trade
            # in this 15m bar. Do NOT suppress on a bare bar-boundary crossing.
            if self._bos_taken_bar == current_15m_bar:
                ctx.bos_detected_15m  = False
                ctx.choch_detected_15m = False

            result = make_decision(ctx)
            action = result.action

            # Track T2 funnel rejections
            if self._t1_stopped_at_loss and action == Action.HOLD and "T2" not in active_trades:
                reason = result.reason
                if reason.startswith("REASON_"):
                    gate_funnel[reason] = gate_funnel.get(reason, 0) + 1

            # Track BOS/CHoCH events for the CSV/MD reports
            if ctx.bos_detected_15m or ctx.choch_detected_15m:
                event_type = "BOS" if ctx.bos_detected_15m else "CHoCH"
                direction  = "bull" if (ctx.bos_up_15m or ctx.choch_up_15m) else "bear"
                bos_choch_list.append({
                    "timestamp": ts,
                    "event":     event_type,
                    "direction": direction,
                    "session":   ctx.session or "off",
                    "action":    action.value,
                    "bid":       bid,
                    "ask":       ask
                })

            # Retrieve bar context for SL and Point calculations
            bar_high = _safe_float(row_dict.get("bar_high")) or mid
            bar_low  = _safe_float(row_dict.get("bar_low"))  or mid
            atr_15m  = _safe_float(row_dict.get("atr_15_15m"))
            if atr_15m is None or atr_15m <= 0:
                logger.debug("[BacktestEngine] atr_15_15m missing — skipping tick for SL calc")
                atr_15m = None

            # ── Execute action ────────────────────────────────────────────

            if action == Action.OPEN_T1_LONG and "T1" not in active_trades:
                sl_price = self._calc_t1_sl(row_dict, Position.LONG)
                if sl_price is None or atr_15m is None:
                    print(f"DEBUG SL SKIP T1 LONG: sl={sl_price}, atr={atr_15m}")
                    logger.debug("[BacktestEngine] T1 long skipped: no sweep SL or ATR")
                else:
                    lot = self._size_lot(ask, sl_price, current_equity, trade_type="T1")
                    s1  = TradeState(type="T1", trade_pair_id=_pair_id,
                                     bos_direction="bull")
                    self._open_trade(
                        s1, Position.LONG, ask, ts, spread,
                        sl_price, atr_15m, bar_high, bar_low, lot,
                    )
                    s1.entry_reason = result.reason
                    s1.session      = ctx.session or "asian"
                    active_trades["T1"] = s1
                    # Store BOS context for Gate 3 relay
                    self._bos_direction    = "bull"
                    self._bos_time_ms      = ts_ms
                    self._r_dynamic_at_bos = _safe_int(row_dict.get("r_dynamic"))
                    self._bos_taken_bar    = current_15m_bar
                    self._t1_stopped_at_loss = False
                    _pair_id += 1

            elif action == Action.OPEN_T1_SHORT and "T1" not in active_trades:
                sl_price = self._calc_t1_sl(row_dict, Position.SHORT)
                if sl_price is None or atr_15m is None:
                    print(f"DEBUG SL SKIP T1 SHORT: sl={sl_price}, atr={atr_15m}")
                    logger.debug("[BacktestEngine] T1 short skipped: no sweep SL or ATR")
                else:
                    lot = self._size_lot(bid, sl_price, current_equity, trade_type="T1")
                    s1  = TradeState(type="T1", trade_pair_id=_pair_id,
                                     bos_direction="bear")
                    self._open_trade(
                        s1, Position.SHORT, bid, ts, spread,
                        sl_price, atr_15m, bar_high, bar_low, lot,
                    )
                    s1.entry_reason = result.reason
                    s1.session      = ctx.session or "asian"
                    active_trades["T1"] = s1
                    self._bos_direction    = "bear"
                    self._bos_time_ms      = ts_ms
                    self._r_dynamic_at_bos = _safe_int(row_dict.get("r_dynamic"))
                    self._bos_taken_bar    = current_15m_bar
                    self._t1_stopped_at_loss = False
                    _pair_id += 1

            elif action == Action.OPEN_T2_LONG and "T2" not in active_trades:
                # T2 is a PENDING limit order — not filled until price reaches fvg_mid
                meta      = result.metadata
                fvg_mid   = _safe_float(meta.get("fvg_mid"))
                fvg_low   = _safe_float(meta.get("fvg_low"))
                fvg_high  = _safe_float(meta.get("fvg_high"))
                expire_at = meta.get("expire_at")

                if fvg_mid is None or fvg_low is None:
                    logger.debug("[BacktestEngine] T2 long skipped: no fvg_mid in metadata")
                else:
                    sl_price = fvg_low - (T2_BUFFER_PIPS * self.PIP_SIZE)
                    lot      = self._size_lot(fvg_mid, sl_price, current_equity, trade_type="T2")
                    s2       = TradeState(
                        type="T2", trade_pair_id=_pair_id - 1,
                        position=Position.LONG,
                        pending=True,
                        fvg_mid=fvg_mid,
                        fvg_low=fvg_low,
                        fvg_high=fvg_high,
                        expire_at_ms=expire_at or (ts_ms + T2_TIMEOUT_MS),
                        sl=sl_price,
                        lot=lot,
                        bos_direction="bull",
                    )
                    s2.entry_reason = result.reason
                    s2.session      = ctx.session or "asian"
                    active_trades["T2"] = s2

            elif action == Action.OPEN_T2_SHORT and "T2" not in active_trades:
                meta      = result.metadata
                fvg_mid   = _safe_float(meta.get("fvg_mid"))
                fvg_low   = _safe_float(meta.get("fvg_low"))
                fvg_high  = _safe_float(meta.get("fvg_high"))
                expire_at = meta.get("expire_at")

                if fvg_mid is None or fvg_high is None:
                    logger.debug("[BacktestEngine] T2 short skipped: no fvg_mid in metadata")
                else:
                    sl_price = fvg_high + (T2_BUFFER_PIPS * self.PIP_SIZE)
                    lot      = self._size_lot(fvg_mid, sl_price, current_equity, trade_type="T2")
                    s2       = TradeState(
                        type="T2", trade_pair_id=_pair_id - 1,
                        position=Position.SHORT,
                        pending=True,
                        fvg_mid=fvg_mid,
                        fvg_low=fvg_low,
                        fvg_high=fvg_high,
                        expire_at_ms=expire_at or (ts_ms + T2_TIMEOUT_MS),
                        sl=sl_price,
                        lot=lot,
                        bos_direction="bear",
                    )
                    s2.entry_reason = result.reason
                    s2.session      = ctx.session or "asian"
                    active_trades["T2"] = s2

            elif action == Action.CLOSE_T1 and "T1" in active_trades:
                trade = self._close_trade(active_trades.pop("T1"), bid, ask, ts, "strategy")
                trades.append(trade)
                equity_curve.append(equity_curve[-1] + trade.pnl)

            elif action == Action.CLOSE_T2 and "T2" in active_trades:
                tstate = active_trades.pop("T2")
                if not tstate.pending:
                    trade = self._close_trade(tstate, bid, ask, ts, "strategy")
                    trades.append(trade)
                    equity_curve.append(equity_curve[-1] + trade.pnl)

            elif action == Action.CLOSE_ALL:
                for tid in list(active_trades.keys()):
                    tstate = active_trades.pop(tid)
                    if tstate.pending:
                        continue
                    trade = self._close_trade(tstate, bid, ask, ts, "strategy_all")
                    trades.append(trade)
                    equity_curve.append(equity_curve[-1] + trade.pnl)

        # ── End of data — close all remaining active trades ───────────────
        if active_trades and not ticks_df.empty:
            last = dict(ticks_df.iloc[-1])
            last_bid = _safe_float(last.get("bid")) or 0.0
            last_ask = _safe_float(last.get("ask")) or 0.0
            last_ts  = pd.Timestamp(last["timestamp_utc"]).to_pydatetime()
            for tid in list(active_trades.keys()):
                tstate = active_trades.pop(tid)
                if tstate.pending:
                    continue    # pending orders simply expire at end of data
                trade = self._close_trade(tstate, last_bid, last_ask, last_ts, "end_of_data")
                trades.append(trade)
                equity_curve.append(equity_curve[-1] + trade.pnl)

        return self._compile_results(
            trades, len(ticks_df), equity_curve, gate_funnel, bos_choch_list, thresholds
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Trade helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _open_trade(
        self,
        state:     TradeState,
        direction: Position,
        price:     float,
        ts:        datetime,
        spread:    float,
        sl_price:  float,
        atr_15m:   float,
        bar_high:  float,
        bar_low:   float,
        lot:       float,
    ) -> None:
        """
        Initialise a trade that is already filled (T1, or T2 after limit hit).
        SL must be pre-calculated by the caller (structural, not ATR-generic).
        """
        state.position          = direction
        state.entry_price       = price
        state.entry_time        = ts
        state.entry_spread      = spread
        state.peak_price        = price
        state.pending           = False
        state.sl                = sl_price
        state.lot               = lot
        state.risk_distance     = abs(price - sl_price)
        state.entry_candle_high = bar_high
        state.entry_candle_low  = bar_low

        # Point 1 — structural breakeven trigger
        if direction == Position.LONG:
            state.point1_price = max(bar_high, price + atr_15m * P1_ATR_FACTOR)
        else:
            state.point1_price = min(bar_low,  price - atr_15m * P1_ATR_FACTOR)

        # Point 2 — trail activation; must be at least P2_P1_MIN_GAP × ATR beyond Point 1
        if direction == Position.LONG:
            raw_p2 = price + atr_15m * P2_ATR_FACTOR
            min_p2 = state.point1_price + atr_15m * P2_P1_MIN_GAP
            state.point2_price = max(raw_p2, min_p2)
        else:
            raw_p2 = price - atr_15m * P2_ATR_FACTOR
            min_p2 = state.point1_price - atr_15m * P2_P1_MIN_GAP
            state.point2_price = min(raw_p2, min_p2)

        state.sl_pips = state.risk_distance / self.PIP_SIZE

        logger.debug(
            f"[BacktestEngine] Opened {state.type} {direction.value} @ {price:.5f} "
            f"SL={sl_price:.5f} P1={state.point1_price:.5f} P2={state.point2_price:.5f} "
            f"lot={lot:.4f}"
        )

    def _activate_pending_t2(
        self,
        state:          TradeState,
        fill_price:     float,
        ts:             datetime,
        spread:         float,
        row_dict:       dict,
        current_equity: float,
    ) -> None:
        """
        Transition a pending T2 limit order to active after price hit fvg_mid.
        Recalculates Point 1 / Point 2 from the actual fill price.
        Sets optional structural TP if R:R qualifies.
        """
        atr_15m   = _safe_float(row_dict.get("atr_15_15m"))
        bar_high  = _safe_float(row_dict.get("bar_high")) or fill_price
        bar_low   = _safe_float(row_dict.get("bar_low"))  or fill_price

        if atr_15m is None or atr_15m <= 0:
            logger.warning("[BacktestEngine] T2 fill: atr_15_15m missing")
            atr_15m = state.risk_distance   # fallback — risk_distance was set at pending creation

        self._open_trade(
            state, state.position, fill_price, ts, spread,
            state.sl, atr_15m, bar_high, bar_low, state.lot,
        )

        # Optional structural TP at nearest Tier 1/2 level with qualifying R:R
        tp = self._find_structural_tp(state, row_dict)
        state.tp = tp
        if tp is not None:
            logger.debug(f"[BacktestEngine] T2 structural TP set @ {tp:.5f}")

    def _calc_t1_sl(self, row_dict: dict, direction: Position) -> Optional[float]:
        """
        T1 SL = lowest/highest wick extreme in the sweep zone + T1_BUFFER_PIPS.
        The feature engineer must provide sweep_candle_low / sweep_candle_high
        as the pre-computed extreme across all 15m candles in the sweep zone.
        Falls back to None if not available (trade is skipped).
        """
        buf = T1_BUFFER_PIPS * self.PIP_SIZE
        if direction == Position.LONG:
            low = _safe_float(row_dict.get("sweep_candle_low"))
            if low is None:
                return None
            return low - buf
        else:
            high = _safe_float(row_dict.get("sweep_candle_high"))
            if high is None:
                return None
            return high + buf

    def _find_structural_tp(
        self, state: TradeState, row_dict: dict
    ) -> Optional[float]:
        """
        T2 optional TP at nearest Tier 1/2 liquidity level in trade direction,
        only if R:R ≥ MIN_TP_RR.
        Tier 1/2 levels available in row: prev_day_high, prev_day_low,
        current_session_high, current_session_low, prev_session_high, prev_session_low.
        """
        entry = state.entry_price
        sl    = state.sl
        if entry is None or sl is None:
            return None

        risk = abs(entry - sl)
        if risk <= 0:
            return None

        # Collect all Tier 1/2 levels
        level_keys = [
            "prev_day_high", "prev_day_low",
            "current_session_high", "current_session_low",
            "prev_session_high",    "prev_session_low",
        ]
        candidates: List[float] = []
        for key in level_keys:
            val = _safe_float(row_dict.get(key))
            if val is None:
                continue
            if state.position == Position.LONG and entry is not None and val > entry:
                candidates.append(val)
            elif state.position == Position.SHORT and entry is not None and val < entry:
                candidates.append(val)

        if not candidates:
            return None

        nearest = min(candidates, key=lambda p: abs(p - entry))
        reward  = abs(nearest - entry)

        if reward >= risk * MIN_TP_RR:
            return nearest
        return None

    def _size_lot(
        self,
        entry:          float,
        sl:             float,
        equity:         float,
        trade_type:     str = "T1",
    ) -> float:
        """
        Lot sizing to respect the 1% combined max loss rule.
        For T1: allocate up to 0.5% risk (half of 1% budget for the pair).
        For T2: allocate the remaining 0.5%.
        Dollar risk = equity × risk_fraction.
        Lot = dollar_risk / (risk_pips × pip_value_per_lot).

        These fractions can be tuned — the key constraint is T1+T2 ≤ 1%.
        """
        risk_fraction = MAX_ACCOUNT_RISK / 2.0   # 0.5% per leg
        dollar_risk   = equity * risk_fraction
        risk_pips     = abs(entry - sl) / self.PIP_SIZE
        if risk_pips <= 0:
            return 0.01   # minimum fallback
        lot = dollar_risk / (risk_pips * self.LOT_PIP_VALUE)
        # Round to 2 decimal places (standard lot increment for retail brokers)
        lot = max(0.01, round(lot, 2))
        return lot

    def _update_fractal_trail(self, state: TradeState, row_dict: dict) -> None:
        """
        Fractal trailing SL — updates SL to confirmed 1m swing extremes.
        One-way only: SL can only move in the direction of the trade, never widen.

        The feature engineer must provide:
          confirmed_1m_swing_low  — most recent confirmed 1m swing low price
          confirmed_1m_swing_high — most recent confirmed 1m swing high price

        These must be confirmed (L=R=3 lag satisfied) — NOT the live forming bar.
        """
        buf = TRAIL_BUFFER_PIPS * self.PIP_SIZE

        if state.position == Position.LONG:
            swing_low = _safe_float(row_dict.get("confirmed_1m_swing_low"))
            if swing_low is not None:
                candidate = swing_low - buf
                if candidate > state.sl:          # one-way only — never widen
                    state.sl = candidate
                    logger.debug(
                        f"[BacktestEngine] T{state.type} trail SL raised to {state.sl:.5f}"
                    )

        elif state.position == Position.SHORT:
            swing_high = _safe_float(row_dict.get("confirmed_1m_swing_high"))
            if swing_high is not None:
                candidate = swing_high + buf
                if candidate < state.sl:          # one-way only — never widen
                    state.sl = candidate
                    logger.debug(
                        f"[BacktestEngine] T{state.type} trail SL lowered to {state.sl:.5f}"
                    )

    def _close_trade(
        self,
        state:  TradeState,
        bid:    float,
        ask:    float,
        ts:     datetime,
        reason: str,
    ) -> CompletedTrade:
        exit_price = bid if state.position == Position.LONG else ask
        pnl_raw    = (
            exit_price - state.entry_price
            if state.position == Position.LONG
            else state.entry_price - exit_price
        )
        pnl_pips   = pnl_raw / self.PIP_SIZE
        # Dollar PnL = price movement × lot size × pip value per lot / pip size
        pnl_dollar = pnl_pips * state.lot * self.LOT_PIP_VALUE
        hold_time  = (ts - state.entry_time).total_seconds() / 60.0

        return CompletedTrade(
            symbol           = self.symbol,
            type             = state.type,
            trade_pair_id    = state.trade_pair_id,
            direction        = state.position,
            entry_time       = state.entry_time,
            exit_time        = ts,
            entry_price      = state.entry_price,
            exit_price       = exit_price,
            pnl              = pnl_dollar,
            pnl_pips         = pnl_pips,
            pnl_raw          = pnl_raw,
            lot              = state.lot,
            exit_reason      = reason,
            entry_reason     = state.entry_reason,
            session          = state.session,
            point1_hit       = state.point1_hit,
            point2_hit       = state.point2_hit,
            hold_time_m      = hold_time,
            sl_pips_at_entry = state.sl_pips,
        )

    def _is_wide_spread(self, spread: float) -> bool:
        return spread >= self.WIDE_SPREAD_PIPS * self.PIP_SIZE

    def _unrealised_pnl(self, state: TradeState, mid: float) -> float:
        if state.position == Position.FLAT or state.entry_price is None:
            return 0.0
        if state.position == Position.LONG:
            return (mid - state.entry_price) * state.lot * self.LOT_PIP_VALUE / self.PIP_SIZE
        return (state.entry_price - mid) * state.lot * self.LOT_PIP_VALUE / self.PIP_SIZE

    # ─────────────────────────────────────────────────────────────────────────
    # Results
    # ─────────────────────────────────────────────────────────────────────────

    def _compile_results(
        self,
        trades:         List[CompletedTrade],
        total_ticks:    int,
        equity:         List[float],
        gate_funnel:    Dict[str, int],
        bos_choch_list: List[dict] = [],
        thresholds:     dict       = {},
    ) -> BacktestResult:
        if not trades:
            return BacktestResult(
                total_ticks = total_ticks,
                total_trades = 0, winning_trades = 0, losing_trades = 0,
                gross_pnl = 0.0, win_rate = 0.0, avg_pnl_per_trade = 0.0,
                max_drawdown = 0.0, profit_factor = 0.0, sharpe_ratio = 0.0,
                gate_funnel = gate_funnel,
                bos_choch_events = bos_choch_list,
                thresholds = thresholds
            )

        wins       = [t for t in trades if t.pnl > 0]
        losses     = [t for t in trades if t.pnl <= 0]
        gross_pnl  = sum(t.pnl for t in trades)
        win_gross  = sum(t.pnl for t in wins)
        loss_gross = sum(t.pnl for t in losses)

        t1_trades = [t for t in trades if t.type == "T1"]
        t2_trades = [t for t in trades if t.type == "T2"]

        session_stats: Dict[str, dict] = {}
        for sess in ("asian", "london", "newyork"):
            s_trades = [t for t in trades if t.session == sess]
            session_stats[sess] = {
                "total": len(s_trades),
                "wins":  len([t for t in s_trades if t.pnl > 0]),
            }

        wr = len(wins) / len(trades)
        pf = (win_gross / abs(loss_gross)) if loss_gross < 0 else float("inf")

        max_dd   = self._calc_max_drawdown(equity)
        sharpe   = self._calc_sharpe(trades)

        return BacktestResult(
            total_ticks       = total_ticks,
            total_trades      = len(trades),
            winning_trades    = len(wins),
            losing_trades     = len(losses),
            gross_pnl         = gross_pnl,
            win_rate          = wr,
            avg_pnl_per_trade = gross_pnl / len(trades),
            max_drawdown      = max_dd,
            profit_factor     = pf,
            sharpe_ratio      = sharpe,
            trades            = trades,
            t1_wins           = len([t for t in t1_trades if t.pnl > 0]),
            t1_losses         = len([t for t in t1_trades if t.pnl <= 0]),
            t2_wins           = len([t for t in t2_trades if t.pnl > 0]),
            t2_losses         = len([t for t in t2_trades if t.pnl <= 0]),
            session_stats     = session_stats,
            gate_funnel       = gate_funnel,
            bos_choch_events  = bos_choch_list,
            thresholds        = thresholds,
        )

    # ── Report Export ────────────────────────────────────────────────────────
    
    def save_reports(self, result: BacktestResult, basename: str) -> None:
        """Save both Markdown summary and Filtered CSV event log."""
        from pathlib import Path
        Path("reports").mkdir(exist_ok=True)
        
        md_path  = f"reports/{basename}.md"
        csv_path = f"reports/{basename}_events.csv"
        
        self._write_markdown_report(result, md_path)
        self._write_csv_report(result, csv_path)
        
        logger.info(f"[BacktestEngine] Reports saved to {md_path} and {csv_path}")

    def _write_markdown_report(self, result: BacktestResult, path: str) -> None:
        """Generates a premium Markdown summary of the backtest."""
        lines = [
            f"# Backtest Report: {self.symbol}",
            f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Core Statistics",
            "| Metric | Value |",
            "| :--- | :--- |",
            f"| **Total Trades** | {result.total_trades} |",
            f"| **Win Rate** | {result.win_rate:.1%} |",
            f"| **Gross PnL** | {result.gross_pnl:+.2f} |",
            f"| **Profit Factor** | {result.profit_factor:.2f} |",
            f"| **Sharpe Ratio** | {result.sharpe_ratio:.3f} |",
            f"| **Max Drawdown** | {result.max_drawdown:.2f} |",
            "",
            "## Strategy Breakdown",
            "| Phase | Trades | Wins | Losses | Win Rate |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        
        t1_total = result.t1_wins + result.t1_losses
        t1_wr = f"{result.t1_wins/t1_total:.1%}" if t1_total > 0 else "n/a"
        lines.append(f"| T1 (Scout) | {t1_total} | {result.t1_wins} | {result.t1_losses} | {t1_wr} |")
        
        t2_total = result.t2_wins + result.t2_losses
        t2_wr = f"{result.t2_wins/t2_total:.1%}" if t2_total > 0 else "n/a"
        lines.append(f"| T2 (Sniper) | {t2_total} | {result.t2_wins} | {result.t2_losses} | {t2_wr} |")
        
        lines += [
            "",
            "## Current Setup (Thresholds)",
            "| Parameter | Value |",
            "| :--- | :--- |",
        ]
        for k, v in result.thresholds.items():
            lines.append(f"| {k} | {v} |")
        
        lines += [
            "",
            "## BOS/CHoCH Detection Log",
            "| Timestamp | Event | Direction | Session | Action Taken |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        # Show last 50 events to keep MD readable
        events_to_show = list(result.bos_choch_events)[-50:]
        for ev in events_to_show:
            lines.append(
                f"| {ev['timestamp']} | {ev['event']} | {ev['direction']} | {ev['session']} | {ev['action']} |"
            )
        
        if len(result.bos_choch_events) > 50:
            lines.append(f"\n*... showing last 50 of {len(result.bos_choch_events)} total events. Full log in CSV.*")

        with open(path, "w") as f:
            f.write("\n".join(lines))

    def _write_csv_report(self, result: BacktestResult, path: str) -> None:
        """Writes a filtered CSV containing only BOS/CHoCH event ticks."""
        if not result.bos_choch_events:
            logger.warning("[BacktestEngine] No events to write to CSV")
            return
            
        df = pd.DataFrame(result.bos_choch_events)
        df.to_csv(path, index=False)

    @staticmethod
    def _calc_max_drawdown(equity: List[float]) -> float:
        """
        Maximum peak-to-trough drawdown in dollar terms from the equity curve.
        """
        if len(equity) < 2:
            return 0.0
        arr     = np.array(equity, dtype=float)
        peak    = np.maximum.accumulate(arr)
        dd      = peak - arr
        return float(np.max(dd))

    @staticmethod
    def _calc_sharpe(trades: List[CompletedTrade], risk_free: float = 0.0) -> float:
        """
        Annualised Sharpe ratio from per-trade PnL returns.
        Uses trade count as the time unit (trade-based Sharpe).
        """
        if len(trades) < 2:
            return 0.0
        returns = np.array([t.pnl for t in trades], dtype=float)
        mean    = np.mean(returns) - risk_free
        std     = np.std(returns, ddof=1)
        if std == 0:
            return 0.0
        # Annualise assuming ~252 trades per year as a convention
        return float((mean / std) * math.sqrt(252))


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(val) -> Optional[float]:
    try:
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