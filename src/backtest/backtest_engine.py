"""
Backtesting Engine — Event-Based Skeleton
Iterates through enriched tick data from DuckDBStore and fires a tick_event
callback for each tick, allowing complex strategy logic like trailing stops
and spread-aware liquidity gap detection.

Architecture:
    BacktestEngine.run(strategy_fn) →  for each tick → strategy_fn(tick_event)
                                           └── strategy_fn decides: open / close / hold

    strategy_fn(tick_event: TickEvent) → Action | None
        • open_long(stop_loss_pips)
        • open_short(stop_loss_pips)
        • close()
        • hold  (return None)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Optional, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from src.gold.duckdb_store import DuckDBStore

logger = logging.getLogger(__name__)


# ── Enums & Data Classes ─────────────────────────────────────────────────────

class Position(Enum):
    FLAT  = "FLAT"
    LONG  = "LONG"
    SHORT = "SHORT"


class Action(Enum):
    OPEN_LONG   = "OPEN_LONG"
    OPEN_SHORT  = "OPEN_SHORT"
    CLOSE       = "CLOSE"
    HOLD        = "HOLD"


@dataclass
class TickEvent:
    """
    Snapshot of a single tick passed to the strategy callback.
    Contains price, indicators, and the current trade state.
    """
    timestamp_utc:  datetime
    symbol:         str
    bid:            float
    ask:            float
    spread:         float           # ask - bid (pips / price units)
    mid:            float           # (bid + ask) / 2
    volume:         float
    rsi_14:         Optional[float]
    atr_14:         Optional[float]
    liq_level:      Optional[float]
    liq_type:       Optional[str]
    bar_open:       Optional[float]
    bar_high:       Optional[float]
    bar_low:        Optional[float]
    bar_close:      Optional[float]
    # Read-only view of the current trade state
    current_position:   Position
    entry_price:        Optional[float]
    trailing_stop:      Optional[float]
    unrealised_pnl:     float = 0.0


@dataclass
class TradeState:
    """Mutable state for the active (or absent) trade."""
    position:        Position     = Position.FLAT
    entry_price:     Optional[float] = None
    trailing_stop:   Optional[float] = None
    entry_time:      Optional[datetime] = None
    entry_spread:    Optional[float] = None
    peak_price:      Optional[float] = None   # used for trailing stop


@dataclass
class CompletedTrade:
    """Record of a completed round-trip trade."""
    symbol:       str
    direction:    Position
    entry_time:   datetime
    exit_time:    datetime
    entry_price:  float
    exit_price:   float
    pnl:          float
    pnl_pips:     float
    exit_reason:  str    # "trailing_stop" | "strategy" | "end_of_data"


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
    trades:             list[CompletedTrade] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"\n{'='*50}\n"
            f"  BACKTEST RESULT\n"
            f"{'='*50}\n"
            f"  Ticks processed : {self.total_ticks:,}\n"
            f"  Total trades    : {self.total_trades}\n"
            f"  Win / Loss      : {self.winning_trades} / {self.losing_trades}\n"
            f"  Win Rate        : {self.win_rate:.1%}\n"
            f"  Gross PnL       : {self.gross_pnl:+.2f}\n"
            f"  Avg PnL / trade : {self.avg_pnl_per_trade:+.2f}\n"
            f"  Max Drawdown    : {self.max_drawdown:.2f}\n"
            f"{'='*50}"
        )


# ── Engine ────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Event-based backtesting engine.

    Feeds enriched ticks from DuckDB to a user-supplied strategy function
    one tick at a time, managing trade state and performance accounting.

    Usage:
        engine = BacktestEngine(
            store=store,
            symbol="XAUUSD",
            trailing_stop_pips=50,
            gap_threshold_pips=30,
        )
        result = engine.run(
            strategy_fn=my_strategy,
            from_dt=datetime(2024, 1, 1, tzinfo=timezone.utc),
            to_dt=datetime(2024, 12, 31, tzinfo=timezone.utc),
        )
        print(result)
    """

    def __init__(
        self,
        store: "DuckDBStore",
        symbol: str = "XAUUSD",
        trailing_stop_pips: float = 50.0,
        gap_threshold_pips: float = 30.0,
        pip_size: float = 0.01,        # XAUUSD: 1 pip = $0.01
    ):
        self.store              = store
        self.symbol             = symbol
        self.trailing_stop_pips = trailing_stop_pips
        self.gap_threshold_pips = gap_threshold_pips
        self.pip_size           = pip_size

    def load_ticks(self, from_dt: datetime, to_dt: datetime) -> pd.DataFrame:
        """Pull enriched tick features from DuckDB."""
        df = self.store.query_features(self.symbol, from_dt, to_dt)
        if df.empty:
            logger.warning(
                f"[BacktestEngine] No data for {self.symbol} "
                f"between {from_dt} and {to_dt}"
            )
        return df

    def run(
        self,
        strategy_fn: Callable[[TickEvent], Optional[Action]],
        from_dt: Optional[datetime] = None,
        to_dt:   Optional[datetime] = None,
        ticks_df: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """
        Main backtest loop.

        Args:
            strategy_fn : A function that receives a TickEvent and returns
                          an Action (or None / Action.HOLD to do nothing).
            from_dt     : Start datetime for loading ticks from DuckDB.
            to_dt       : End datetime for loading ticks from DuckDB.
            ticks_df    : Optionally supply a pre-loaded DataFrame directly
                          (bypasses DuckDB load — useful for testing).

        Returns:
            BacktestResult with full trade log and summary statistics.
        """
        if ticks_df is None:
            if from_dt is None or to_dt is None:
                raise ValueError("Provide either ticks_df or both from_dt and to_dt")
            ticks_df = self.load_ticks(from_dt, to_dt)

        if ticks_df.empty:
            logger.error("[BacktestEngine] No ticks to process — aborting")
            return BacktestResult(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0)

        state   = TradeState()
        trades:  list[CompletedTrade] = []
        equity_curve: list[float] = [0.0]

        logger.info(f"[BacktestEngine] Starting run over {len(ticks_df):,} ticks")

        for _, row in ticks_df.iterrows():
            bid   = float(row.get("bid",  0.0))
            ask   = float(row.get("ask",  0.0))
            mid   = (bid + ask) / 2.0
            spread = ask - bid
            ts    = pd.Timestamp(row["timestamp_utc"]).to_pydatetime()

            # Build the tick event passed to the strategy
            event = TickEvent(
                timestamp_utc    = ts,
                symbol           = str(row.get("symbol", self.symbol)),
                bid              = bid,
                ask              = ask,
                spread           = spread,
                mid              = mid,
                volume           = float(row.get("volume",    0.0)),
                rsi_14           = _safe_float(row.get("rsi_14")),
                atr_14           = _safe_float(row.get("atr_14")),
                liq_level        = _safe_float(row.get("liq_level")),
                liq_type         = row.get("liq_type"),
                bar_open         = _safe_float(row.get("bar_open")),
                bar_high         = _safe_float(row.get("bar_high")),
                bar_low          = _safe_float(row.get("bar_low")),
                bar_close        = _safe_float(row.get("bar_close")),
                current_position = state.position,
                entry_price      = state.entry_price,
                trailing_stop    = state.trailing_stop,
                unrealised_pnl   = self._unrealised_pnl(state, mid),
            )

            # Check trailing stop BEFORE handing to strategy
            if state.position != Position.FLAT:
                stopped_out = self._check_trailing_stop(state, bid, ask)
                if stopped_out:
                    trade = self._close_trade(state, bid, ask, ts, "trailing_stop")
                    trades.append(trade)
                    equity_curve.append(equity_curve[-1] + trade.pnl)
                    state = TradeState()
                    continue  # skip strategy for this tick

                # Check for spread-aware liquidity gap
                gap_trade = self._check_liquidity_gap(state, event, ts)
                if gap_trade:
                    trades.append(gap_trade)
                    equity_curve.append(equity_curve[-1] + gap_trade.pnl)
                    state = TradeState()
                    continue

                # Update trailing stop
                self._update_trailing_stop(state, bid, ask)

            # Call the strategy
            action = strategy_fn(event)

            if action == Action.OPEN_LONG and state.position == Position.FLAT:
                self._open_trade(state, Position.LONG, ask, ts, spread)

            elif action == Action.OPEN_SHORT and state.position == Position.FLAT:
                self._open_trade(state, Position.SHORT, bid, ts, spread)

            elif action == Action.CLOSE and state.position != Position.FLAT:
                trade = self._close_trade(state, bid, ask, ts, "strategy")
                trades.append(trade)
                equity_curve.append(equity_curve[-1] + trade.pnl)
                state = TradeState()

        # Close any open position at end of data
        if state.position != Position.FLAT and not ticks_df.empty:
            last = ticks_df.iloc[-1]
            trade = self._close_trade(
                state,
                float(last.get("bid", 0.0)),
                float(last.get("ask", 0.0)),
                pd.Timestamp(last["timestamp_utc"]).to_pydatetime(),
                "end_of_data",
            )
            trades.append(trade)
            equity_curve.append(equity_curve[-1] + trade.pnl)

        return self._compile_results(trades, len(ticks_df), equity_curve)

    # ── Trade helpers ────────────────────────────────────────────────────────

    def _open_trade(
        self,
        state: TradeState,
        direction: Position,
        price: float,
        ts: datetime,
        spread: float,
    ) -> None:
        state.position    = direction
        state.entry_price = price
        state.entry_time  = ts
        state.entry_spread = spread
        state.peak_price  = price
        # Initial trailing stop (in price units)
        stop_distance = self.trailing_stop_pips * self.pip_size
        if direction == Position.LONG:
            state.trailing_stop = price - stop_distance
        else:
            state.trailing_stop = price + stop_distance
        logger.debug(f"[BacktestEngine] Opened {direction.value} @ {price:.5f} | TS={state.trailing_stop:.5f}")

    def _close_trade(
        self,
        state: TradeState,
        bid: float,
        ask: float,
        ts: datetime,
        reason: str,
    ) -> CompletedTrade:
        exit_price = bid if state.position == Position.LONG else ask
        pnl_raw = (
            exit_price - state.entry_price
            if state.position == Position.LONG
            else state.entry_price - exit_price
        )
        pnl_pips = pnl_raw / self.pip_size
        logger.debug(
            f"[BacktestEngine] Closed {state.position.value} @ {exit_price:.5f} "
            f"PnL={pnl_raw:+.2f} ({reason})"
        )
        return CompletedTrade(
            symbol      = self.symbol,
            direction   = state.position,
            entry_time  = state.entry_time,
            exit_time   = ts,
            entry_price = state.entry_price,
            exit_price  = exit_price,
            pnl         = pnl_raw,
            pnl_pips    = pnl_pips,
            exit_reason = reason,
        )

    def _update_trailing_stop(self, state: TradeState, bid: float, ask: float) -> None:
        """Ratchet trailing stop as price moves in our favour."""
        stop_distance = self.trailing_stop_pips * self.pip_size
        if state.position == Position.LONG:
            if bid > state.peak_price:
                state.peak_price = bid
                state.trailing_stop = max(
                    state.trailing_stop, bid - stop_distance
                )
        elif state.position == Position.SHORT:
            if ask < state.peak_price:
                state.peak_price = ask
                state.trailing_stop = min(
                    state.trailing_stop, ask + stop_distance
                )

    def _check_trailing_stop(self, state: TradeState, bid: float, ask: float) -> bool:
        """Return True if the trailing stop has been hit."""
        if state.position == Position.LONG and bid <= state.trailing_stop:
            return True
        if state.position == Position.SHORT and ask >= state.trailing_stop:
            return True
        return False

    def _check_liquidity_gap(
        self, state: TradeState, event: TickEvent, ts: datetime
    ) -> Optional[CompletedTrade]:
        """
        Detect a spread-aware liquidity gap: when the spread widens beyond
        gap_threshold_pips, assume price has gapped through our level and
        force-close the position at the worse fill price.
        """
        gap_threshold = self.gap_threshold_pips * self.pip_size
        if event.spread >= gap_threshold and state.position != Position.FLAT:
            logger.debug(
                f"[BacktestEngine] Liquidity gap detected: spread={event.spread:.5f} "
                f"threshold={gap_threshold:.5f} — force-closing"
            )
            return self._close_trade(state, event.bid, event.ask, ts, "liquidity_gap")
        return None

    def _unrealised_pnl(self, state: TradeState, mid: float) -> float:
        if state.position == Position.FLAT or state.entry_price is None:
            return 0.0
        if state.position == Position.LONG:
            return mid - state.entry_price
        return state.entry_price - mid

    # ── Results compilation ──────────────────────────────────────────────────

    def _compile_results(
        self,
        trades: list[CompletedTrade],
        total_ticks: int,
        equity_curve: list[float],
    ) -> BacktestResult:
        if not trades:
            return BacktestResult(total_ticks, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, [])

        gross_pnl    = sum(t.pnl for t in trades)
        winners      = [t for t in trades if t.pnl > 0]
        losers       = [t for t in trades if t.pnl <= 0]
        win_rate     = len(winners) / len(trades) if trades else 0.0
        avg_pnl      = gross_pnl / len(trades)

        # Max drawdown from equity curve
        peak = equity_curve[0]
        max_dd = 0.0
        for e in equity_curve:
            if e > peak:
                peak = e
            dd = peak - e
            if dd > max_dd:
                max_dd = dd

        return BacktestResult(
            total_ticks      = total_ticks,
            total_trades     = len(trades),
            winning_trades   = len(winners),
            losing_trades    = len(losers),
            gross_pnl        = gross_pnl,
            win_rate         = win_rate,
            avg_pnl_per_trade = avg_pnl,
            max_drawdown     = max_dd,
            trades           = trades,
        )


# ── Utility ───────────────────────────────────────────────────────────────────

def _safe_float(val) -> Optional[float]:
    """Convert a possibly NaN / None value to a Python float or None."""
    try:
        f = float(val)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None
