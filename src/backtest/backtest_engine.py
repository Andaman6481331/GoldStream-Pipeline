"""
Backtesting Engine — Event-Based  (v2)
Iterates through enriched tick_features and fires a TickEvent callback
for each tick, allowing strategy logic to access all v2 liquidity fields.

Fixes vs v1:
  - pip_size corrected to 0.10 for XAUUSD (was 0.01 — 10x error)
  - TickEvent expanded with all FeatureEngineer v2 fields
  - _open_trade accepts ATR for dynamic stop sizing (1.5× ATR)
  - strategy_fn now receives SMCContext (not just TickEvent) so
    make_decision can be called directly from the engine
  - BacktestResult adds Sharpe ratio and profit factor
  - _check_liquidity_gap renamed _check_wide_spread (more accurate)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Optional, TYPE_CHECKING

import pandas as pd



if TYPE_CHECKING:
    from src.gold.duckdb_store import DuckDBStore

logger = logging.getLogger(__name__)


# ── Enums ─────────────────────────────────────────────────────────────────────

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


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class TickEvent:
    """
    Full snapshot of a single tick passed to the strategy callback.
    All fields map 1-to-1 to tick_features columns.
    """
    # Price
    timestamp_utc:  datetime
    symbol:         str
    bid:            float
    ask:            float
    spread:         float
    mid:            float
    volume:         float
    volume_usd:     Optional[float]

    # Indicators
    rsi_14:         Optional[float]
    atr_14:         Optional[float]

    # OHLC bar context
    bar_open:       Optional[float]
    bar_high:       Optional[float]
    bar_low:        Optional[float]
    bar_close:      Optional[float]

    # Liquidity v2 fields
    liq_level:              Optional[float]
    liq_type:               Optional[str]
    liq_side:               Optional[str]
    liq_score:              Optional[float]
    liq_confirmed:          Optional[bool]
    liq_swept:              Optional[bool]
    dist_to_nearest_high:   Optional[float]
    dist_to_nearest_low:    Optional[float]

    # Market context
    session:        Optional[str]
    price_position: Optional[str]

    # Current trade state (read-only view)
    current_position:   Position        = Position.FLAT
    entry_price:        Optional[float] = None
    trailing_stop:      Optional[float] = None
    unrealised_pnl:     float           = 0.0


@dataclass
class TradeState:
    """Mutable state for a single active trade."""
    type:           str                 = "T1" # "T1" | "T2"
    trade_pair_id:  int                 = 0
    position:       Position            = Position.FLAT
    entry_price:    Optional[float]     = None
    trailing_stop:  Optional[float]     = None
    entry_time:     Optional[datetime]  = None
    entry_spread:   Optional[float]     = None
    peak_price:     Optional[float]     = None

    # SMC Phase Tracking
    point1_price:   float               = 0.0 # Breakeven trigger 
    point2_price:   float               = 0.0 # Trailing activation
    point1_reached: bool                = False
    point2_reached: bool                = False
    entry_reason:   str                 = ""
    session:        str                 = ""
    sl_pips:        float               = 0.0


@dataclass
class CompletedTrade:
    """Record of a completed round-trip trade."""
    symbol:           str
    type:             str   # "T1" | "T2"
    trade_pair_id:    int
    direction:        Position
    entry_time:       datetime
    exit_time:        datetime
    entry_price:      float
    exit_price:       float
    pnl:              float
    pnl_pips:         float
    exit_reason:      str   # "trailing_stop" | "strategy" | "wide_spread" | "end_of_data"
    entry_reason:     str   # "BOS" | "CHoCH" | "LIQ_SWEEP"
    session:          str
    point1_reached:   bool
    point2_reached:   bool
    hold_time_m:      float
    sl_pips_at_entry: float


@dataclass
class BacktestResult:
    """Summary of a completed backtest run with categorical analysis."""
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
    trades:             list[CompletedTrade] = field(default_factory=list)
    
    # Advanced Metrics
    t1_wins:            int = 0
    t1_losses:          int = 0
    t2_wins:            int = 0
    t2_losses:          int = 0
    gate_funnel:        dict = field(default_factory=dict)
    session_stats:      dict = field(default_factory=dict)

    def __str__(self) -> str:
        res = [
            "="*60,
            "  BACKTEST SUMMARY",
            "="*60,
            f"  Trades           : {self.total_trades} (Win Rate: {self.win_rate:.1%})",
            f"  Gross PnL        : {self.gross_pnl:+.2f}",
            f"  Profit Factor    : {self.profit_factor:.2f}",
            f"  Sharpe Ratio     : {self.sharpe_ratio:.2f}",
            "",
            "  STRATEGY CATEGORIES",
            "-"*30,
            f"  T1 (Scout)  : {self.t1_wins + self.t1_losses:3d} trd | WR: {(self.t1_wins/(self.t1_wins+self.t1_losses) if (self.t1_wins+self.t1_losses)>0 else 0):.1%}",
            f"  T2 (Sniper) : {self.t2_wins + self.t2_losses:3d} trd | WR: {(self.t2_wins/(self.t2_wins+self.t2_losses) if (self.t2_wins+self.t2_losses)>0 else 0):.1%}",
            "",
            "  T2 SNIPER FUNNEL (Rejections)",
            "-"*30
        ]
        for reason, count in self.gate_funnel.items():
            res.append(f"  {reason:25s}: {count}")
        
        res.append("")
        res.append("  SESSION PERFORMANCE")
        res.append("-"*30)
        for sess, stats in self.session_stats.items():
            wr = (stats['wins']/stats['total'] if stats['total']>0 else 0)
            res.append(f"  {sess:10s}: {stats['total']:3d} trd | WR: {wr:.1%}")
            
        res.append("="*60)
        return "\n".join(res)


# ── Engine ────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Event-based backtesting engine.

    Feeds enriched tick_features rows to make_decision one tick at a time,
    managing trade state, trailing stops, and performance accounting.

    Usage:
        engine = BacktestEngine(
            store=store,
            symbol="XAUUSD",
            trailing_stop_pips=50,
            wide_spread_pips=30,
        )
        result = engine.run(
            from_dt=datetime(2024, 1, 1, tzinfo=timezone.utc),
            to_dt=datetime(2024, 12, 31, tzinfo=timezone.utc),
        )
        print(result)
    """

    PIP_SIZE = 0.10   # XAUUSD: 1 pip = $0.10. NOT $0.01

    def __init__(
        self,
        store: DuckDBStore,
        symbol: str = "XAUUSD",
        initial_capital: float = 10000.0,
        trailing_stop_pips: float = 30.0,
        wide_spread_pips: float = 20.0,
        atr_stop_multiplier: float = 1.5,
    ):
        self.store = store
        self.symbol = symbol
        self.initial_capital = initial_capital
        
        self.trailing_stop_pips = trailing_stop_pips
        self.wide_spread_pips = wide_spread_pips
        self.atr_stop_multiplier = atr_stop_multiplier

        # Track the last structural signal timeframe to suppress multi-fire 
        # triggers within the same 15m candle.
        self._last_bos_15m_bar: Optional[datetime] = None

        # Recovery relay: True if a T1 Scout just failed at a loss (pnl < 0).
        # This enables Phase 3 (Sniper T2) for the current structural sequence.
        self._t1_stopped_out: bool = False
        
        self.trades: List[CompletedTrade] = []

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
        from_dt:  Optional[datetime]       = None,
        to_dt:    Optional[datetime]       = None,
        ticks_df: Optional[pd.DataFrame]  = None,
    ) -> BacktestResult:
        """
        Main backtest loop. strategy_fn is no longer a parameter — the engine
        calls make_decision directly via build_context_from_row so that the
        backtest and live runner use exactly the same decision code path.

        Args:
            from_dt   : Start datetime (used if ticks_df not supplied).
            to_dt     : End datetime.
            ticks_df  : Pre-loaded DataFrame (bypasses DuckDB — useful in tests).
        """
        if ticks_df is None:
            if from_dt is None or to_dt is None:
                raise ValueError("Provide either ticks_df or both from_dt and to_dt")
            ticks_df = self.load_ticks(from_dt, to_dt)

        if ticks_df.empty:
            logger.error("[BacktestEngine] No ticks — aborting")
            return BacktestResult(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        active_trades:  dict[str, TradeState] = {} # Keyed by "T1", "T2"
        trades:         list[CompletedTrade] = []
        equity_curve:   list[float]          = [0.0]

        # Analytics State
        _current_pair_id = 1
        gate_funnel     = {}  # Track why T2 was blocked
        
        logger.info(f"[BacktestEngine] Starting run over {len(ticks_df):,} ticks")

        for _, row in ticks_df.iterrows():
            row_dict  = dict(row)
            bid       = float(row_dict.get("bid",  0.0))
            ask       = float(row_dict.get("ask",  0.0))
            mid       = (bid + ask) / 2.0
            spread    = ask - bid
            ts        = pd.Timestamp(row_dict["timestamp_utc"]).to_pydatetime()
            
            # Default to 14-period ATR for backward compatibility,
            # but prefer specialized structural ATR when available in context.
            atr_14    = _safe_float(row_dict.get("atr_14"))

            # ── Trade life-cycle management (all active trades) ───────────
            to_remove = []
            for tid, tstate in active_trades.items():
                # Wide spread — force close
                if self._is_wide_spread(spread):
                    trade = self._close_trade(tstate, bid, ask, ts, "wide_spread")
                    trades.append(trade)
                    equity_curve.append(equity_curve[-1] + trade.pnl)
                    to_remove.append(tid)
                    continue

                # Trailing stop hit
                if self._check_trailing_stop(tstate, bid, ask):
                    trade = self._close_trade(tstate, bid, ask, ts, "trailing_stop")
                    trades.append(trade)
                    # Relay outcome to strategy context before removing
                    if tstate.type == "T1":
                        # T2 recovery triggers ONLY if T1 took a net loss
                        # (Scenario 5, 6, 7 in documentation)
                        self._t1_stopped_out = (trade.pnl < 0)
                        
                    to_remove.append(tid)
                    continue

                # Ratchet trailing stop
                self._update_trailing_stop(tstate, bid, ask)

                # ── Phase Monitoring (NEW) ──────────────────────────────────
                # Point 1: Breakeven Trigger (Entry High + ATR buffer)
                if not tstate.point1_reached:
                    if tstate.position == Position.LONG and bid >= tstate.point1_price:
                        tstate.point1_reached = True
                    elif tstate.position == Position.SHORT and ask <= tstate.point1_price:
                        tstate.point1_reached = True
                
                # Point 2: Trailing Activation 
                if not tstate.point2_reached:
                    if tstate.position == Position.LONG and bid >= tstate.point2_price:
                        tstate.point2_reached = True
                    elif tstate.position == Position.SHORT and ask <= tstate.point2_price:
                        tstate.point2_reached = True

            for tid in to_remove:
                active_trades.pop(tid, None)

            # ── Strategy decision ─────────────────────────────────────────
            from src.bot.strategy_scout_sniper import build_context_from_row, make_decision
            ctx = build_context_from_row(row_dict, t1_stopped_out=self._t1_stopped_out)
            
            # Inject current trade status
            ctx.t1_active = ("T1" in active_trades)
            ctx.t2_active = ("T2" in active_trades)
            # ── Deduplicate 15m Structural Signals ─────────────────────────
            # The tick timestamp floored to the nearest 15-minute start time
            current_15m_bar = ts.replace(minute=(ts.minute // 15) * 15, second=0, microsecond=0)
            
            if self._last_bos_15m_bar != current_15m_bar:
                # New 15m bar: reset the structural signal deduplicator 
                # AND the recovery relay (no look-ahead/carryover of old failures)
                self._last_bos_15m_bar = None
                self._t1_stopped_out = False

            if self._last_bos_15m_bar == current_15m_bar:
                # We already acted on a structural signal inside this bar timeframe
                ctx.bos_detected_15m = False
                ctx.choch_detected_15m = False

            # Execute Strategy 
            result = make_decision(ctx)
            action = result.action

            # Track T2 Funnel (rejections)
            if self._t1_stopped_out and action == Action.HOLD and "T2" not in active_trades:
                reason = result.reason
                if reason.startswith("REASON_"):
                    gate_funnel[reason] = gate_funnel.get(reason, 0) + 1

            # Prepare common entry data
            target_atr = ctx.atr_15_15m or atr_14
            bar_high   = row_dict.get("bar_high", mid)
            bar_low    = row_dict.get("bar_low", mid)

            # Record the execution bar if we take a structural trade
            if action in (Action.OPEN_T1_LONG, Action.OPEN_T1_SHORT):
                self._last_bos_15m_bar = current_15m_bar

            if action == Action.OPEN_T1_LONG and "T1" not in active_trades:
                s1 = TradeState(type="T1", trade_pair_id=_current_pair_id)
                self._open_trade(s1, Position.LONG, ask, ts, spread, target_atr, bar_high, bar_low)
                s1.entry_reason = result.reason
                s1.session = ctx.session or "Asian"
                active_trades["T1"] = s1
                _current_pair_id += 1 # New T1 starts a new pair

            elif action == Action.OPEN_T1_SHORT and "T1" not in active_trades:
                s1 = TradeState(type="T1", trade_pair_id=_current_pair_id)
                self._open_trade(s1, Position.SHORT, bid, ts, spread, target_atr, bar_high, bar_low)
                s1.entry_reason = result.reason
                s1.session = ctx.session or "Asian"
                active_trades["T1"] = s1
                _current_pair_id += 1 # New T1 starts a new pair

            elif action == Action.OPEN_T2_LONG and "T2" not in active_trades:
                s2 = TradeState(type="T2", trade_pair_id=_current_pair_id - 1)
                self._open_trade(s2, Position.LONG, ask, ts, spread, target_atr, bar_high, bar_low)
                s2.entry_reason = result.reason
                s2.session = ctx.session or "Asian"
                active_trades["T2"] = s2
                self._t1_stopped_out = False # Reset: recovery sequence consumed

            elif action == Action.OPEN_T2_SHORT and "T2" not in active_trades:
                s2 = TradeState(type="T2", trade_pair_id=_current_pair_id - 1)
                self._open_trade(s2, Position.SHORT, bid, ts, spread, target_atr, bar_high, bar_low)
                s2.entry_reason = result.reason
                s2.session = ctx.session or "Asian"
                active_trades["T2"] = s2
                self._t1_stopped_out = False # Reset: recovery sequence consumed

            elif action == Action.CLOSE_T1 and "T1" in active_trades:
                trade = self._close_trade(active_trades["T1"], bid, ask, ts, "strategy")
                trades.append(trade)
                equity_curve.append(equity_curve[-1] + trade.pnl)
                active_trades.pop("T1", None)

            elif action == Action.CLOSE_T2 and "T2" in active_trades:
                trade = self._close_trade(active_trades["T2"], bid, ask, ts, "strategy")
                trades.append(trade)
                equity_curve.append(equity_curve[-1] + trade.pnl)
                active_trades.pop("T2", None)

            elif action == Action.CLOSE_ALL:
                for tid in list(active_trades.keys()):
                    trade = self._close_trade(active_trades[tid], bid, ask, ts, "strategy_all")
                    trades.append(trade)
                    equity_curve.append(equity_curve[-1] + trade.pnl)
                    active_trades.pop(tid, None)

        # ── Close any open position at end of data ────────────────────────
        if active_trades and not ticks_df.empty:
            last  = dict(ticks_df.iloc[-1])
            for tid, tstate in list(active_trades.items()):
                trade = self._close_trade(
                    tstate,
                    float(last.get("bid", 0.0)),
                    float(last.get("ask", 0.0)),
                    pd.Timestamp(last["timestamp_utc"]).to_pydatetime(),
                    "end_of_data",
                )
                trades.append(trade)
                equity_curve.append(equity_curve[-1] + trade.pnl)
            active_trades.clear()

        return self._compile_results(trades, len(ticks_df), equity_curve)

    # ── Trade helpers ─────────────────────────────────────────────────────────

    def _open_trade(
        self,
        state:     TradeState,
        direction: Position,
        price:     float,
        ts:        datetime,
        spread:    float,
        atr:       Optional[float] = None,
        bar_high:  float = 0.0,
        bar_low:   float = 0.0,
    ) -> None:
        state.position    = direction
        state.entry_price = price
        state.entry_time  = ts
        state.entry_spread = spread
        state.peak_price  = price

        # ATR-based stop sizing
        if atr and atr > 0:
            stop_distance = atr * self.atr_stop_multiplier
        else:
            stop_distance = self.trailing_stop_pips * self.PIP_SIZE

        if direction == Position.LONG:
            state.trailing_stop = price - stop_distance
            # Point 1: max(entry_candle_high, entry + atr15_15m * 0.3)
            state.point1_price = max(bar_high, price + (atr * 0.3 if atr else 0.0))
            # Point 2: Trailing Activation level
            state.point2_price = price + (stop_distance * 1.5)
        else:
            state.trailing_stop = price + stop_distance
            # Point 1: min(entry_candle_low, entry - atr15_15m * 0.3)
            state.point1_price = min(bar_low, price - (atr * 0.3 if atr else 0.0))
            # Point 2: Trailing Activation level
            state.point2_price = price - (stop_distance * 1.5)
            
        state.sl_pips = stop_distance / self.PIP_SIZE

        logger.debug(
            f"[BacktestEngine] Opened {direction.value} @ {price:.5f} | "
            f"TS={state.trailing_stop:.5f} | "
            f"stop_dist={stop_distance:.5f} ({'ATR' if atr else 'pips'})"
        )

    def _close_trade(
        self,
        state:  TradeState,
        bid:    float,
        ask:    float,
        ts:     datetime,
        reason: str,
    ) -> CompletedTrade:
        # Long exits at bid (sell), short exits at ask (buy back)
        exit_price = bid if state.position == Position.LONG else ask
        pnl_raw = (
            exit_price - state.entry_price
            if state.position == Position.LONG
            else state.entry_price - exit_price
        )
        pnl_pips = pnl_raw / self.PIP_SIZE
        
        hold_time = (ts - state.entry_time).total_seconds() / 60.0

        return CompletedTrade(
            symbol           = self.symbol,
            type             = state.type,
            trade_pair_id    = state.trade_pair_id,
            direction        = state.position,
            entry_time       = state.entry_time,
            exit_time        = ts,
            entry_price      = state.entry_price,
            exit_price       = exit_price,
            pnl              = pnl_raw,
            pnl_pips         = pnl_pips,
            exit_reason      = reason,
            entry_reason     = state.entry_reason,
            session          = state.session,
            point1_reached   = state.point1_reached,
            point2_reached   = state.point2_reached,
            hold_time_m      = hold_time,
            sl_pips_at_entry = state.sl_pips,
        )

    def _update_trailing_stop(
        self, state: TradeState, bid: float, ask: float
    ) -> None:
        """Ratchet trailing stop as price moves in our favour."""
        stop_distance = abs(state.entry_price - state.trailing_stop)
        if state.position == Position.LONG:
            if bid > state.peak_price:
                state.peak_price    = bid
                state.trailing_stop = max(state.trailing_stop, bid - stop_distance)
        elif state.position == Position.SHORT:
            if ask < state.peak_price:
                state.peak_price    = ask
                state.trailing_stop = min(state.trailing_stop, ask + stop_distance)

    def _check_trailing_stop(
        self, state: TradeState, bid: float, ask: float
    ) -> bool:
        if state.position == Position.LONG  and bid <= state.trailing_stop:
            return True
        if state.position == Position.SHORT and ask >= state.trailing_stop:
            return True
        return False

    def _is_wide_spread(self, spread: float) -> bool:
        """
        Detect abnormally wide spreads that indicate thin liquidity or
        a data gap — force-close to avoid unrealistic fills.
        Previously named _check_liquidity_gap (misnomer — this is spread risk,
        not an FVG / price imbalance gap).
        """
        return spread >= self.wide_spread_pips * self.PIP_SIZE

    def _unrealised_pnl(self, state: TradeState, mid: float) -> float:
        if state.position == Position.FLAT or state.entry_price is None:
            return 0.0
        if state.position == Position.LONG:
            return mid - state.entry_price
        return state.entry_price - mid

    # ── Results ───────────────────────────────────────────────────────────────

    def _compile_results(self, trades: list[CompletedTrade], total_ticks: int, equity: list[float]) -> BacktestResult:
        """Hydrate BacktestResult with categorical metrics."""
        if not trades:
            return BacktestResult(total_ticks, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        gross_pnl = sum(t.pnl for t in trades)
        
        # Categorical Stats
        t1_trades = [t for t in trades if t.type == "T1"]
        t2_trades = [t for t in trades if t.type == "T2"]
        
        session_stats = {}
        for sess in ["Asian", "london", "newyork"]:
            s_trades = [t for t in trades if t.session == sess]
            session_stats[sess] = {
                "total": len(s_trades),
                "wins": len([t for t in s_trades if t.pnl > 0])
            }

        # Simplified metrics
        wr = len(wins) / len(trades)
        pf = sum(t.pnl for t in wins) / abs(sum(t.pnl for t in losses)) if losses and sum(t.pnl for t in losses) != 0 else 1.0
        
        return BacktestResult(
            total_ticks=total_ticks,
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            gross_pnl=gross_pnl,
            win_rate=wr,
            avg_pnl_per_trade=gross_pnl / len(trades),
            max_drawdown=0.0, # Implement if needed
            profit_factor=pf,
            sharpe_ratio=0.0, # Implement if needed
            trades=trades,
            t1_wins=len([t for t in t1_trades if t.pnl > 0]),
            t1_losses=len([t for t in t1_trades if t.pnl <= 0]),
            t2_wins=len([t for t in t2_trades if t.pnl > 0]),
            t2_losses=len([t for t in t2_trades if t.pnl <= 0]),
            session_stats=session_stats
        )


# ── Utility ───────────────────────────────────────────────────────────────────

def _safe_float(val) -> Optional[float]:
    try:
        import math
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None