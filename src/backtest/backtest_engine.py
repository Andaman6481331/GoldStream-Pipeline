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

from src.bot.strategy import make_decision, build_context_from_row, Action

if TYPE_CHECKING:
    from src.gold.duckdb_store import DuckDBStore

logger = logging.getLogger(__name__)


# ── Enums ─────────────────────────────────────────────────────────────────────

class Position(Enum):
    FLAT  = "FLAT"
    LONG  = "LONG"
    SHORT = "SHORT"


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
    """Mutable state for the active (or absent) trade."""
    position:       Position            = Position.FLAT
    entry_price:    Optional[float]     = None
    trailing_stop:  Optional[float]     = None
    entry_time:     Optional[datetime]  = None
    entry_spread:   Optional[float]     = None
    peak_price:     Optional[float]     = None


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
    exit_reason:  str   # "trailing_stop" | "strategy" | "wide_spread" | "end_of_data"


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
    profit_factor:      float           # gross_win / gross_loss
    sharpe_ratio:       float           # simplified: mean/std of per-trade PnL
    trades:             list[CompletedTrade] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"\n{'='*55}\n"
            f"  BACKTEST RESULT\n"
            f"{'='*55}\n"
            f"  Ticks processed  : {self.total_ticks:,}\n"
            f"  Total trades     : {self.total_trades}\n"
            f"  Win / Loss       : {self.winning_trades} / {self.losing_trades}\n"
            f"  Win Rate         : {self.win_rate:.1%}\n"
            f"  Gross PnL        : {self.gross_pnl:+.2f}\n"
            f"  Avg PnL / trade  : {self.avg_pnl_per_trade:+.2f}\n"
            f"  Max Drawdown     : {self.max_drawdown:.2f}\n"
            f"  Profit Factor    : {self.profit_factor:.2f}\n"
            f"  Sharpe Ratio     : {self.sharpe_ratio:.2f}\n"
            f"{'='*55}"
        )


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
        store:                "DuckDBStore",
        symbol:               str   = "XAUUSD",
        trailing_stop_pips:   float = 50.0,
        wide_spread_pips:     float = 30.0,
        atr_stop_multiplier:  float = 1.5,    # stop = 1.5 × ATR when ATR available
    ):
        self.store                = store
        self.symbol               = symbol
        self.trailing_stop_pips   = trailing_stop_pips
        self.wide_spread_pips     = wide_spread_pips
        self.atr_stop_multiplier  = atr_stop_multiplier

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

        state         = TradeState()
        trades:         list[CompletedTrade] = []
        equity_curve:   list[float]          = [0.0]

        logger.info(f"[BacktestEngine] Starting run over {len(ticks_df):,} ticks")

        for _, row in ticks_df.iterrows():
            row_dict  = dict(row)
            bid       = float(row_dict.get("bid",  0.0))
            ask       = float(row_dict.get("ask",  0.0))
            mid       = (bid + ask) / 2.0
            spread    = ask - bid
            ts        = pd.Timestamp(row_dict["timestamp_utc"]).to_pydatetime()
            atr       = _safe_float(row_dict.get("atr_14"))

            # ── Trailing stop check BEFORE strategy ───────────────────────
            if state.position != Position.FLAT:

                # Wide spread — force close (slippage protection)
                if self._is_wide_spread(spread):
                    trade = self._close_trade(state, bid, ask, ts, "wide_spread")
                    trades.append(trade)
                    equity_curve.append(equity_curve[-1] + trade.pnl)
                    state = TradeState()
                    continue

                # Trailing stop hit
                if self._check_trailing_stop(state, bid, ask):
                    trade = self._close_trade(state, bid, ask, ts, "trailing_stop")
                    trades.append(trade)
                    equity_curve.append(equity_curve[-1] + trade.pnl)
                    state = TradeState()
                    continue

                # Ratchet trailing stop upward/downward
                self._update_trailing_stop(state, bid, ask)

            # ── Strategy decision ─────────────────────────────────────────
            ctx    = build_context_from_row(row_dict)
            action = make_decision(ctx)

            if action == Action.OPEN_LONG and state.position == Position.FLAT:
                self._open_trade(state, Position.LONG, ask, ts, spread, atr)

            elif action == Action.OPEN_SHORT and state.position == Position.FLAT:
                self._open_trade(state, Position.SHORT, bid, ts, spread, atr)

            elif action == Action.CLOSE and state.position != Position.FLAT:
                trade = self._close_trade(state, bid, ask, ts, "strategy")
                trades.append(trade)
                equity_curve.append(equity_curve[-1] + trade.pnl)
                state = TradeState()

        # ── Close any open position at end of data ────────────────────────
        if state.position != Position.FLAT and not ticks_df.empty:
            last  = dict(ticks_df.iloc[-1])
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

    # ── Trade helpers ─────────────────────────────────────────────────────────

    def _open_trade(
        self,
        state:     TradeState,
        direction: Position,
        price:     float,
        ts:        datetime,
        spread:    float,
        atr:       Optional[float] = None,
    ) -> None:
        state.position    = direction
        state.entry_price = price
        state.entry_time  = ts
        state.entry_spread = spread
        state.peak_price  = price

        # ATR-based stop sizing — more accurate than fixed pip distance
        if atr and atr > 0:
            stop_distance = atr * self.atr_stop_multiplier
        else:
            stop_distance = self.trailing_stop_pips * self.PIP_SIZE

        if direction == Position.LONG:
            state.trailing_stop = price - stop_distance
        else:
            state.trailing_stop = price + stop_distance

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
        logger.debug(
            f"[BacktestEngine] Closed {state.position.value} @ {exit_price:.5f} | "
            f"PnL={pnl_raw:+.4f} ({pnl_pips:+.1f} pips) | {reason}"
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

    def _compile_results(
        self,
        trades:       list[CompletedTrade],
        total_ticks:  int,
        equity_curve: list[float],
    ) -> BacktestResult:
        if not trades:
            return BacktestResult(total_ticks, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [])

        import statistics

        pnls      = [t.pnl for t in trades]
        gross_pnl = sum(pnls)
        winners   = [p for p in pnls if p > 0]
        losers    = [p for p in pnls if p <= 0]
        win_rate  = len(winners) / len(trades)
        avg_pnl   = gross_pnl / len(trades)

        # Profit factor: gross wins / gross losses (>1 = profitable)
        gross_win  = sum(winners) if winners else 0.0
        gross_loss = abs(sum(losers)) if losers else 1e-9
        profit_factor = gross_win / gross_loss

        # Simplified Sharpe: mean PnL / std PnL (no risk-free rate)
        if len(pnls) > 1:
            mean_pnl = statistics.mean(pnls)
            std_pnl  = statistics.stdev(pnls)
            sharpe   = (mean_pnl / std_pnl) if std_pnl > 0 else 0.0
        else:
            sharpe = 0.0

        # Max drawdown from equity curve
        peak   = equity_curve[0]
        max_dd = 0.0
        for e in equity_curve:
            peak  = max(peak, e)
            max_dd = max(max_dd, peak - e)

        return BacktestResult(
            total_ticks       = total_ticks,
            total_trades      = len(trades),
            winning_trades    = len(winners),
            losing_trades     = len(losers),
            gross_pnl         = gross_pnl,
            win_rate          = win_rate,
            avg_pnl_per_trade = avg_pnl,
            max_drawdown      = max_dd,
            profit_factor     = profit_factor,
            sharpe_ratio      = sharpe,
            trades            = trades,
        )


# ── Utility ───────────────────────────────────────────────────────────────────

def _safe_float(val) -> Optional[float]:
    try:
        import math
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None