"""
Gold Layer — FeatureEngineer  (Scout & Sniper — Phase 1 SMC Upgrade)
────────────────────────────────────────────────────────────────────
Existing v3 pipeline (5-min primary TF, liquidity, BOS/CHoCH, FVG) is
preserved exactly.  Phase 1 SMC additions are clearly marked NEW-P1.

Phase 1 additions
─────────────────
1.  Multi-timeframe candle builder
        Internal DataFrames for 1m, 15m, and 4H built in parallel with
        the existing 5m pipeline.  Not exposed as output columns — used
        as computation substrates for downstream features.

2.  SMC-specific ATR indicators
        atr_20_1m   — ATR(20) on 1m candles  (FVG size filter, displacement)
        atr_15_15m  — ATR(15) on 15m candles (R_dynamic, sweep tolerance, P1/P2)
        Both merged to tick resolution alongside existing atr_14 (5m ATR(14)).

3.  Session level tracker
        Snapshots Previous Day High/Low at midnight UTC.
        Snapshots session open High/Low at each session boundary
        (Asian 00:00, London 08:00, NY 13:00 UTC).
        Carries the most recent snapshot forward to every tick via
        forward-fill merge.
        New tick columns:
            prev_day_high, prev_day_low
            current_session_high, current_session_low
            prev_session_high, prev_session_low
            session_boundary  (bool — True on the tick that opened a new session)

4.  15m confirmed swing history
        _build_swing_history_15m() walks closed 15m candles and emits
        a list of SwingPoint(price, bar_time, kind) for confirmed swing
        highs and lows using Williams Fractal logic (L=5, R=R_dynamic).
        Stored in self.swing_highs_15m / self.swing_lows_15m after each
        call to build_features().
        The lists are the direct input required by the Phase 2 structural
        node state machine (HH/LL/StrongLow/StrongHigh).
        Per-tick column added:
            n_confirmed_swing_highs_15m  (int) — count at that point in time
            n_confirmed_swing_lows_15m   (int) — count at that point in time
            (full SwingPoint objects available on self for Phase 2 use)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timezone
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

if TYPE_CHECKING:
    from src.gold.duckdb_store import DuckDBStore

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

# Existing (unchanged)
TIMEFRAMES: list[str]   = ["5min", "1h", "4h", "1d"]
RSI_PERIOD: int         = 14
ATR_PERIOD: int         = 14
BASE_SWING_WINDOW: int  = 5
MIN_SWING_ATR_RATIO     = 0.5
CONFLUENCE_TOLERANCE    = 0.002
SWEEP_TOLERANCE         = 0.0005
N_NEAREST_LEVELS        = 3
MAX_FVG_AGE_BARS        = 20

# NEW-P1: SMC-specific ATR periods
ATR_PERIOD_1M:  int = 20   # atr_20_1m  — FVG size filter, displacement check
ATR_PERIOD_15M: int = 15   # atr_15_15m — R_dynamic, sweep tolerance, P1/P2

# NEW-P1: Williams Fractal parameters for 15m swing history
FRACTAL_L: int = 5         # lookback window — fixed
FRACTAL_R_MAX: int = 5     # maximum forward confirmation window (R_dynamic clamp max)
FRACTAL_R_MIN: int = 2     # minimum forward confirmation window (R_dynamic clamp min)

# NEW-P1: R_dynamic calibration — k = R_DYNAMIC_K_FACTOR × avg_atr_15m
# At average ATR, R_dynamic resolves to 3 (midpoint of 2-5 range)
R_DYNAMIC_K_FACTOR: float = 3.0

# NEW-P1: Session boundary times (UTC hour)
SESSION_BOUNDARIES: dict[str, int] = {
    "asian":  0,
    "london": 8,
    "ny":     13,
}

# Session windows for existing session label (unchanged)
SESSIONS: dict[str, tuple[int, int]] = {
    "london":   (7,  16),
    "new_york": (12, 21),
    "asian":    (0,   7),
    "killzone": (7,   9),
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class SwingPoint:
    """
    A single confirmed Williams Fractal swing point on the 15m chart.
    Confirmed means R candles have closed after the pivot candle.
    """
    price:    float
    bar_time: pd.Timestamp
    kind:     Literal["high", "low"]   # "high" = swing high, "low" = swing low
    bar_idx:  int                       # index in the 15m candle DataFrame


@dataclass
class SessionSnapshot:
    """
    Price levels captured at a session boundary.
    Carried forward to every tick until the next boundary.
    """
    snapshot_time:        pd.Timestamp
    session_name:         str
    prev_day_high:        float
    prev_day_low:         float
    current_session_high: float          # reset to open price at session start
    current_session_low:  float          # reset to open price at session start
    prev_session_high:    float
    prev_session_low:     float


# ── Main class ────────────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Computes professional-grade SMC features from a DataFrame of UnifiedTick rows.

    Phase 1 SMC state (available after build_features()):
        self.swing_highs_15m  list[SwingPoint]  — confirmed 15m swing highs
        self.swing_lows_15m   list[SwingPoint]  — confirmed 15m swing lows
        self.atr_15m_avg      float             — average ATR(15) on 15m
                                                  (use to compute k for R_dynamic)

    Usage
    -----
        fe = FeatureEngineer()
        enriched_df = fe.build_features(unified_ticks_df)
        fe.save_to_duckdb(enriched_df, store)

        # Phase 2 inputs (available immediately after build_features):
        swing_highs = fe.swing_highs_15m
        swing_lows  = fe.swing_lows_15m
    """

    def __init__(self) -> None:
        # NEW-P1: Phase 1 state — populated by build_features()
        self.swing_highs_15m: list[SwingPoint] = []
        self.swing_lows_15m:  list[SwingPoint] = []
        self.atr_15m_avg:     float = 0.0
        self.atr_1m_avg:      float = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def build_features(self, ticks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Full Scout & Sniper feature pipeline.

        Existing steps 1-10 (v3) preserved exactly.
        New Phase 1 SMC steps inserted at appropriate points.

        Returns tick-level DataFrame with all v3 columns plus:
            atr_20_1m, atr_15_15m                    (SMC ATR indicators)
            prev_day_high, prev_day_low               (session level tracker)
            current_session_high, current_session_low
            prev_session_high, prev_session_low
            session_boundary                          (bool)
            n_confirmed_swing_highs_15m               (count at each tick)
            n_confirmed_swing_lows_15m
        """
        if ticks_df.empty:
            logger.warning("[FeatureEngineer] Empty DataFrame — skipping")
            return ticks_df

        ticks_df = ticks_df.copy()
        ticks_df["timestamp_utc"] = pd.to_datetime(ticks_df["timestamp_utc"], utc=True)
        ticks_df = ticks_df.sort_values("timestamp_utc")
        ticks_df["mid"] = (ticks_df["bid"] + ticks_df["ask"]) / 2.0

        # ── Step 1 & 2: existing 5-min candles + indicators (unchanged) ──────
        candles_5m = self._resample_ohlc(ticks_df, "5min")
        candles_5m = self._compute_indicators(candles_5m)

        # ── NEW-P1 Step A: build SMC timeframe candles ────────────────────────
        candles_1m  = self._resample_ohlc(ticks_df, "1min")
        candles_15m = self._resample_ohlc(ticks_df, "15min")
        candles_4h  = self._resample_ohlc(ticks_df, "4h")

        # ── NEW-P1 Step B: SMC ATR indicators ────────────────────────────────
        candles_1m  = self._compute_smc_atr(candles_1m,  period=ATR_PERIOD_1M,  col="atr_20_1m")
        candles_15m = self._compute_smc_atr(candles_15m, period=ATR_PERIOD_15M, col="atr_15_15m")

        # Store averages for Phase 2 R_dynamic calibration
        self.atr_15m_avg = float(candles_15m["atr_15_15m"].median())
        self.atr_1m_avg  = float(candles_1m["atr_20_1m"].median())

        # ── NEW-P1 Step C: session level tracker ─────────────────────────────
        session_levels_df = self._build_session_levels(ticks_df, candles_4h)

        # ── NEW-P1 Step D: 15m confirmed swing history ────────────────────────
        self.swing_highs_15m, self.swing_lows_15m = self._build_swing_history_15m(
            candles_15m
        )
        swing_count_df = self._build_swing_count_series(
            ticks_df, candles_15m, self.swing_highs_15m, self.swing_lows_15m
        )

        # ── Step 3: multi-TF liquidity levels (unchanged) ────────────────────
        all_liq: list[pd.DataFrame] = []
        for tf in TIMEFRAMES:
            candles_tf = self._resample_ohlc(ticks_df, tf)
            if len(candles_tf) < BASE_SWING_WINDOW * 2 + 1:
                logger.debug(f"[FeatureEngineer] Not enough candles for TF={tf}, skipping")
                continue
            candles_tf = self._compute_indicators(candles_tf)
            liq_tf = self._identify_liquidity_levels(candles_tf, tf)
            if not liq_tf.empty:
                all_liq.append(liq_tf)

        liq_df = pd.concat(all_liq, ignore_index=True) if all_liq else pd.DataFrame()

        # ── Step 4: swept levels (unchanged) ─────────────────────────────────
        if not liq_df.empty:
            liq_df = self._mark_swept_levels(liq_df, candles_5m)

        # ── Step 5: confluence scoring (unchanged) ───────────────────────────
        if not liq_df.empty:
            liq_df = self._score_confluence(liq_df)

        # ── Step 6: BOS / CHoCH on 5-min candles (unchanged) ─────────────────
        candles_5m = self._compute_structure_breaks(candles_5m)

        # ── Step 7: FVG detection + fill tracking (unchanged) ────────────────
        fvg_df = self._compute_fvg(candles_5m)
        if not fvg_df.empty:
            fvg_df = self._mark_filled_fvgs(fvg_df, candles_5m)

        # ── Step 8: merge everything to tick resolution ───────────────────────
        enriched = self._merge_to_ticks(ticks_df, candles_5m, liq_df, fvg_df)

        # ── NEW-P1 Step E: merge SMC features to tick resolution ──────────────
        enriched = self._merge_smc_atrs(enriched, candles_1m, candles_15m)
        enriched = self._merge_session_levels(enriched, session_levels_df)
        enriched = self._merge_swing_counts(enriched, swing_count_df)

        # ── Step 9 & 10: session + price position (unchanged) ────────────────
        enriched = self._add_session_label(enriched)
        enriched = self._add_price_position(enriched)

        logger.info(
            f"[FeatureEngineer] Built features: {len(enriched)} ticks, "
            f"{len(candles_5m)} 5-min candles, "
            f"{len(candles_1m)} 1-min candles, "
            f"{len(candles_15m)} 15-min candles, "
            f"{len(candles_4h)} 4H candles, "
            f"{len(liq_df)} liquidity levels, "
            f"{len(fvg_df)} FVGs, "
            f"{len(self.swing_highs_15m)} confirmed 15m swing highs, "
            f"{len(self.swing_lows_15m)} confirmed 15m swing lows"
        )
        return enriched

    def save_to_duckdb(self, df: pd.DataFrame, store: "DuckDBStore") -> None:
        """Persist enriched tick DataFrame to the Gold layer DuckDB store."""
        if df.empty:
            logger.warning("[FeatureEngineer] Nothing to save — DataFrame is empty")
            return
        cols = [
            # ── Identity ──────────────────────────────────────────────────────
            "timestamp_utc", "symbol", "bid", "ask", "mid",
            "volume", "volume_usd", "source",
            # ── 5-min candle (unchanged) ──────────────────────────────────────
            "bar_open", "bar_high", "bar_low", "bar_close",
            # ── Indicators (unchanged) ────────────────────────────────────────
            "rsi_14", "atr_14",
            # ── Liquidity (unchanged) ─────────────────────────────────────────
            "liq_level", "liq_type", "liq_side", "liq_tf",
            "liq_score", "liq_confirmed", "liq_swept",
            "dist_to_nearest_high", "dist_to_nearest_low",
            # ── Structure (unchanged) ─────────────────────────────────────────
            "structure_direction", "bos_detected", "choch_detected",
            # ── FVG (unchanged) ───────────────────────────────────────────────
            "fvg_high", "fvg_low", "fvg_side",
            "fvg_timestamp", "fvg_filled", "fvg_age_bars",
            # ── Context (unchanged) ───────────────────────────────────────────
            "price_position", "session",
            # ── NEW-P1: SMC ATR indicators ────────────────────────────────────
            "atr_20_1m",
            "atr_15_15m",
            # ── NEW-P1: Session levels ────────────────────────────────────────
            "prev_day_high",
            "prev_day_low",
            "current_session_high",
            "current_session_low",
            "prev_session_high",
            "prev_session_low",
            "session_boundary",
            # ── NEW-P1: Swing history counts ──────────────────────────────────
            "n_confirmed_swing_highs_15m",
            "n_confirmed_swing_lows_15m",
        ]
        available = [c for c in cols if c in df.columns]
        store.upsert_features(df[available])

    # =========================================================================
    # NEW-P1: SMC ATR indicators
    # =========================================================================

    @staticmethod
    def _compute_smc_atr(
        candles: pd.DataFrame, period: int, col: str
    ) -> pd.DataFrame:
        """
        Compute ATR with a specific period and store in a named column.
        Used for atr_20_1m and atr_15_15m — separate from existing atr_14.
        Requires at least `period` closed candles to produce a non-NaN value.
        """
        candles = candles.copy()
        if len(candles) < period:
            candles[col] = np.nan
            return candles

        candles[col] = AverageTrueRange(
            high=candles["bar_high"],
            low=candles["bar_low"],
            close=candles["bar_close"],
            window=period,
        ).average_true_range()
        return candles

    def _merge_smc_atrs(
        self,
        enriched:    pd.DataFrame,
        candles_1m:  pd.DataFrame,
        candles_15m: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Forward-fill atr_20_1m and atr_15_15m from their respective candle
        DataFrames onto the tick-level enriched DataFrame.

        Uses merge_asof (backward direction) — each tick gets the ATR value
        of the most recently closed candle on that timeframe.
        """
        enriched = enriched.copy()

        # 1m ATR
        if "atr_20_1m" in candles_1m.columns:
            atr_1m = (
                candles_1m[["bar_time", "atr_20_1m"]]
                .dropna(subset=["atr_20_1m"])
                .sort_values("bar_time")
            )
            enriched = pd.merge_asof(
                enriched.sort_values("timestamp_utc"),
                atr_1m,
                left_on="timestamp_utc",
                right_on="bar_time",
                direction="backward",
            ).drop(columns=["bar_time"], errors="ignore")
        else:
            enriched["atr_20_1m"] = np.nan

        # 15m ATR
        if "atr_15_15m" in candles_15m.columns:
            atr_15m = (
                candles_15m[["bar_time", "atr_15_15m"]]
                .dropna(subset=["atr_15_15m"])
                .sort_values("bar_time")
            )
            enriched = pd.merge_asof(
                enriched.sort_values("timestamp_utc"),
                atr_15m,
                left_on="timestamp_utc",
                right_on="bar_time",
                direction="backward",
            ).drop(columns=["bar_time"], errors="ignore")
        else:
            enriched["atr_15_15m"] = np.nan

        return enriched

    # =========================================================================
    # NEW-P1: Session level tracker
    # =========================================================================

    def _build_session_levels(
        self,
        ticks_df:   pd.DataFrame,
        candles_4h: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Walk forward through ticks and snapshot liquidity levels at each
        session boundary (daily midnight + session opens).

        Returns a DataFrame with one row per boundary event, forward-filled
        onto ticks by _merge_session_levels().

        Columns:
            snapshot_time         — UTC timestamp of the boundary
            session_boundary      — always True (marks the boundary tick)
            prev_day_high         — high of the previous completed day
            prev_day_low          — low of the previous completed day
            current_session_high  — opening price of this session (resets here)
            current_session_low   — opening price of this session (resets here)
            prev_session_high     — high of the session that just closed
            prev_session_low      — low of the session that just closed

        Logic
        ─────
        Daily snapshot (midnight UTC):
            prev_day_high/low = high/low of the day candle that just closed.
            Derived from 4H candles to avoid needing a 1D feed.

        Session snapshot (London 08:00, NY 13:00, Asian 00:00 UTC):
            current_session_high/low resets to the opening mid price.
            prev_session_high/low = high/low of the session that just ended.
            The running session high/low is maintained by scanning ticks
            since the last session boundary.
        """
        ticks_df = ticks_df.copy().sort_values("timestamp_utc")
        dates = ticks_df["timestamp_utc"].dt.date.unique()

        snapshots: list[dict] = []

        # Running state
        prev_day_high:    float = np.nan
        prev_day_low:     float = np.nan
        prev_session_high: float = np.nan
        prev_session_low:  float = np.nan
        current_session_high: float = np.nan
        current_session_low:  float = np.nan
        current_session_name: str = "asian"
        session_start_time: Optional[pd.Timestamp] = None

        # Build a quick lookup: date → daily high/low from 4H candles
        daily_hl = self._compute_daily_hl_from_4h(candles_4h)

        boundary_hours = sorted(SESSION_BOUNDARIES.values())   # [0, 8, 13]

        for _, tick in ticks_df.iterrows():
            ts: pd.Timestamp = tick["timestamp_utc"]
            mid: float = tick["mid"]
            hour: int = ts.hour
            date = ts.date()

            # Check if we've crossed a session boundary
            if session_start_time is None:
                # First tick ever — initialise silently
                session_start_time = ts
                current_session_high = mid
                current_session_low  = mid
                continue

            # Determine which session boundary this hour belongs to
            new_session = self._hour_to_session(hour)
            if new_session != current_session_name:
                # --- Boundary crossed ---

                # 1. Snapshot the session that just ended
                prev_session_high = current_session_high
                prev_session_low  = current_session_low

                # 2. If crossing midnight (asian boundary at 00:00) →
                #    update prev_day_high/low from yesterday's 4H candles
                if hour == SESSION_BOUNDARIES["asian"]:
                    yesterday = (ts - pd.Timedelta(days=1)).date()
                    if yesterday in daily_hl:
                        prev_day_high = daily_hl[yesterday]["high"]
                        prev_day_low  = daily_hl[yesterday]["low"]

                # 3. Reset current session to this tick's price
                current_session_high = mid
                current_session_low  = mid
                current_session_name = new_session
                session_start_time   = ts

                snapshots.append({
                    "snapshot_time":        ts,
                    "session_boundary":     True,
                    "prev_day_high":        prev_day_high,
                    "prev_day_low":         prev_day_low,
                    "current_session_high": current_session_high,
                    "current_session_low":  current_session_low,
                    "prev_session_high":    prev_session_high,
                    "prev_session_low":     prev_session_low,
                })

            else:
                # Within session — update running high/low
                if mid > current_session_high:
                    current_session_high = mid
                if mid < current_session_low:
                    current_session_low = mid

        if not snapshots:
            # No boundaries found — return empty with correct columns
            return pd.DataFrame(columns=[
                "snapshot_time", "session_boundary",
                "prev_day_high", "prev_day_low",
                "current_session_high", "current_session_low",
                "prev_session_high", "prev_session_low",
            ])

        df = pd.DataFrame(snapshots)
        df["snapshot_time"] = pd.to_datetime(df["snapshot_time"], utc=True)
        return df.sort_values("snapshot_time")

    @staticmethod
    def _compute_daily_hl_from_4h(candles_4h: pd.DataFrame) -> dict:
        """
        Aggregate 4H candles into daily high/low.
        Returns: { date: { "high": float, "low": float } }
        """
        if candles_4h.empty:
            return {}

        candles_4h = candles_4h.copy()
        candles_4h["date"] = pd.to_datetime(candles_4h["bar_time"], utc=True).dt.date

        daily = (
            candles_4h.groupby("date")
            .agg(high=("bar_high", "max"), low=("bar_low", "min"))
            .to_dict(orient="index")
        )
        return daily

    @staticmethod
    def _hour_to_session(hour: int) -> str:
        """Map UTC hour to session name based on boundary hours."""
        if hour >= SESSION_BOUNDARIES["ny"]:
            return "ny"
        if hour >= SESSION_BOUNDARIES["london"]:
            return "london"
        return "asian"

    def _merge_session_levels(
        self,
        enriched:         pd.DataFrame,
        session_levels_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Forward-fill session level snapshots onto tick-level enriched DataFrame.
        Each tick carries the most recently snapshotted levels.
        Ticks before the first snapshot get NaN values.
        """
        enriched = enriched.copy()

        if session_levels_df.empty:
            for col in [
                "prev_day_high", "prev_day_low",
                "current_session_high", "current_session_low",
                "prev_session_high", "prev_session_low",
                "session_boundary",
            ]:
                enriched[col] = np.nan
            enriched["session_boundary"] = False
            return enriched

        merged = pd.merge_asof(
            enriched.sort_values("timestamp_utc"),
            session_levels_df.sort_values("snapshot_time"),
            left_on="timestamp_utc",
            right_on="snapshot_time",
            direction="backward",
        ).drop(columns=["snapshot_time"], errors="ignore")

        # session_boundary should only be True on the exact boundary tick,
        # not forward-filled. Reset all to False then mark boundary ticks.
        merged["session_boundary"] = False

        # Mark ticks that land within 1 second of a snapshot boundary
        if not session_levels_df.empty:
            boundary_times = pd.to_datetime(
                session_levels_df["snapshot_time"].values, utc=True
            )
            for bt in boundary_times:
                mask = (
                    (merged["timestamp_utc"] >= bt) &
                    (merged["timestamp_utc"] < bt + pd.Timedelta(seconds=1))
                )
                merged.loc[mask, "session_boundary"] = True

        return merged

    # =========================================================================
    # NEW-P1: 15m confirmed swing history
    # =========================================================================

    def _build_swing_history_15m(
        self, candles_15m: pd.DataFrame
    ) -> tuple[list[SwingPoint], list[SwingPoint]]:
        """
        Walk all closed 15m candles and emit confirmed Williams Fractal
        swing highs and swing lows.

        Williams Fractal parameters:
            L = FRACTAL_L = 5  (fixed lookback)
            R = R_dynamic      (volatility-adaptive forward confirmation window)
                             = clamp(round(k / atr_15_15m), R_MIN, R_MAX)
                             where k = R_DYNAMIC_K_FACTOR × avg_atr_15_15m

        A swing high at index i is confirmed when:
            candles[i].high > max(highs[i-L : i])          (left side)
            candles[i].high > max(highs[i+1 : i+R+1])      (right side)

        A swing low at index i is confirmed when:
            candles[i].low < min(lows[i-L : i])
            candles[i].low < min(lows[i+1 : i+R+1])

        Note on lag: a swing at index i can only be confirmed after i+R
        candles have closed. This function processes all available closed
        candles but only emits confirmed swings (i.e. the last R candles
        cannot produce new confirmations yet — they are pending).

        Returns:
            (swing_highs, swing_lows) — lists of SwingPoint, sorted by bar_time
        """
        if len(candles_15m) < FRACTAL_L + FRACTAL_R_MAX + 1:
            logger.debug("[FeatureEngineer] Not enough 15m candles for swing detection")
            return [], []

        candles_15m = candles_15m.copy().reset_index(drop=True)
        n = len(candles_15m)

        highs  = candles_15m["bar_high"].values
        lows   = candles_15m["bar_low"].values
        times  = candles_15m["bar_time"].values

        # Compute R_dynamic per candle
        avg_atr = self.atr_15m_avg
        if avg_atr == 0 or np.isnan(avg_atr):
            # Fallback: estimate from price range
            avg_atr = (highs.max() - lows.min()) * 0.005
            logger.debug(
                f"[FeatureEngineer] atr_15m_avg unavailable — using fallback {avg_atr:.4f}"
            )

        atr_col = "atr_15_15m"
        k = R_DYNAMIC_K_FACTOR * avg_atr

        def r_for_index(i: int) -> int:
            if atr_col in candles_15m.columns:
                atr_val = candles_15m[atr_col].iloc[i]
                if pd.isna(atr_val) or atr_val == 0:
                    atr_val = avg_atr
            else:
                atr_val = avg_atr
            r = int(round(k / atr_val))
            return max(FRACTAL_R_MIN, min(FRACTAL_R_MAX, r))

        swing_highs: list[SwingPoint] = []
        swing_lows:  list[SwingPoint] = []

        # Walk every possible pivot candle
        # A pivot at index i requires:
        #   left:  i >= L
        #   right: i + R_dynamic + 1 <= n  (i.e. R candles have closed after i)
        for i in range(FRACTAL_L, n):
            r = r_for_index(i)

            # Right side must be fully confirmed
            if i + r >= n:
                # Not enough candles to the right yet — swing is pending
                continue

            left_highs = highs[i - FRACTAL_L : i]
            left_lows  = lows[i - FRACTAL_L : i]
            right_highs = highs[i + 1 : i + r + 1]
            right_lows  = lows[i + 1 : i + r + 1]

            # Swing High
            if (highs[i] > left_highs.max() and
                    highs[i] > right_highs.max()):
                swing_highs.append(SwingPoint(
                    price    = round(float(highs[i]), 5),
                    bar_time = pd.Timestamp(times[i], tz="UTC"),
                    kind     = "high",
                    bar_idx  = i,
                ))

            # Swing Low
            if (lows[i] < left_lows.min() and
                    lows[i] < right_lows.min()):
                swing_lows.append(SwingPoint(
                    price    = round(float(lows[i]), 5),
                    bar_time = pd.Timestamp(times[i], tz="UTC"),
                    kind     = "low",
                    bar_idx  = i,
                ))

        logger.debug(
            f"[FeatureEngineer] Swing history: "
            f"{len(swing_highs)} highs, {len(swing_lows)} lows "
            f"from {n} 15m candles"
        )
        return swing_highs, swing_lows

    def _build_swing_count_series(
        self,
        ticks_df:     pd.DataFrame,
        candles_15m:  pd.DataFrame,
        swing_highs:  list[SwingPoint],
        swing_lows:   list[SwingPoint],
    ) -> pd.DataFrame:
        """
        For each 15m candle bar_time, compute how many confirmed swing
        highs and lows existed at that point in time.

        Returns a DataFrame with columns:
            bar_time
            n_confirmed_swing_highs_15m
            n_confirmed_swing_lows_15m

        This is merged onto ticks via forward-fill so every tick carries
        the swing count that was current at that moment.

        The count at bar_time T = number of SwingPoints with
        bar_time <= T (i.e. confirmed by the candle at T or earlier).
        """
        if candles_15m.empty:
            return pd.DataFrame(columns=[
                "bar_time",
                "n_confirmed_swing_highs_15m",
                "n_confirmed_swing_lows_15m",
            ])

        bar_times = pd.to_datetime(
            candles_15m["bar_time"].values, utc=True
        )

        high_times = sorted([sh.bar_time for sh in swing_highs])
        low_times  = sorted([sl.bar_time for sl in swing_lows])

        counts = []
        for bt in bar_times:
            n_highs = sum(1 for t in high_times if t <= bt)
            n_lows  = sum(1 for t in low_times  if t <= bt)
            counts.append({
                "bar_time": bt,
                "n_confirmed_swing_highs_15m": n_highs,
                "n_confirmed_swing_lows_15m":  n_lows,
            })

        df = pd.DataFrame(counts)
        df["bar_time"] = pd.to_datetime(df["bar_time"], utc=True)
        return df.sort_values("bar_time")

    def _merge_swing_counts(
        self,
        enriched:       pd.DataFrame,
        swing_count_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Forward-fill swing counts from 15m candle resolution onto ticks.
        """
        enriched = enriched.copy()

        if swing_count_df.empty:
            enriched["n_confirmed_swing_highs_15m"] = 0
            enriched["n_confirmed_swing_lows_15m"]  = 0
            return enriched

        merged = pd.merge_asof(
            enriched.sort_values("timestamp_utc"),
            swing_count_df.sort_values("bar_time"),
            left_on="timestamp_utc",
            right_on="bar_time",
            direction="backward",
        ).drop(columns=["bar_time"], errors="ignore")

        merged["n_confirmed_swing_highs_15m"] = (
            merged["n_confirmed_swing_highs_15m"].fillna(0).astype(int)
        )
        merged["n_confirmed_swing_lows_15m"] = (
            merged["n_confirmed_swing_lows_15m"].fillna(0).astype(int)
        )
        return merged

    # =========================================================================
    # Existing v3 methods — unchanged
    # =========================================================================

    def _resample_ohlc(self, ticks_df: pd.DataFrame, freq: str) -> pd.DataFrame:
        ohlc = (
            ticks_df.set_index("timestamp_utc")["mid"]
            .resample(freq)
            .ohlc()
            .dropna()
        )
        vol = (
            ticks_df.set_index("timestamp_utc")["volume"]
            .resample(freq)
            .sum()
        )
        candles = ohlc.join(vol, how="left")
        candles.index.name = "bar_time"
        candles.columns = ["bar_open", "bar_high", "bar_low", "bar_close", "bar_volume"]
        candles = candles.reset_index()
        candles["bar_time"] = pd.to_datetime(candles["bar_time"], utc=True)
        return candles

    def _compute_indicators(self, candles: pd.DataFrame) -> pd.DataFrame:
        candles = candles.copy()
        candles[f"rsi_{RSI_PERIOD}"] = RSIIndicator(
            close=candles["bar_close"], window=RSI_PERIOD
        ).rsi()
        candles[f"atr_{ATR_PERIOD}"] = AverageTrueRange(
            high=candles["bar_high"],
            low=candles["bar_low"],
            close=candles["bar_close"],
            window=ATR_PERIOD,
        ).average_true_range()
        return candles

    def _compute_structure_breaks(self, candles: pd.DataFrame) -> pd.DataFrame:
        candles = candles.copy()
        n = len(candles)

        highs  = candles["bar_high"].values
        lows   = candles["bar_low"].values
        closes = candles["bar_close"].values

        atr_col = f"atr_{ATR_PERIOD}"
        avg_atr = float(candles[atr_col].median()) if atr_col in candles.columns else 0.0
        if avg_atr == 0 or np.isnan(avg_atr):
            avg_atr = (highs.max() - lows.min()) * 0.01

        structure_direction = [None] * n
        bos_detected        = [False] * n
        choch_detected      = [False] * n

        last_confirmed_swing_high: Optional[float] = None
        last_confirmed_swing_low:  Optional[float] = None
        current_direction: Optional[str] = None

        w = BASE_SWING_WINDOW

        for i in range(w, n):
            local_atr = candles[atr_col].iloc[i] if atr_col in candles.columns else avg_atr
            if np.isnan(local_atr) or local_atr == 0:
                local_atr = avg_atr

            for j in range(max(0, i - w * 2), i - w + 1):
                lo_j = j - w
                hi_j = j + w + 1
                if lo_j < 0 or hi_j > n:
                    continue

                if highs[j] >= highs[lo_j:hi_j].max() - 1e-8:
                    if self._confirm_swing_high(closes, lows, j, min(j + 21, i + 1)):
                        if (last_confirmed_swing_high is None
                                or highs[j] > last_confirmed_swing_high):
                            last_confirmed_swing_high = highs[j]

                if lows[j] <= lows[lo_j:hi_j].min() + 1e-8:
                    if self._confirm_swing_low(closes, highs, j, min(j + 21, i + 1)):
                        if (last_confirmed_swing_low is None
                                or lows[j] < last_confirmed_swing_low):
                            last_confirmed_swing_low = lows[j]

            if last_confirmed_swing_high is not None:
                if closes[i] > last_confirmed_swing_high:
                    if current_direction == "bullish":
                        bos_detected[i] = True
                    else:
                        choch_detected[i] = True
                    current_direction = "bullish"
                    last_confirmed_swing_high = None

            if last_confirmed_swing_low is not None:
                if closes[i] < last_confirmed_swing_low:
                    if current_direction == "bearish":
                        bos_detected[i] = True
                    else:
                        choch_detected[i] = True
                    current_direction = "bearish"
                    last_confirmed_swing_low = None

            structure_direction[i] = current_direction

        candles["structure_direction"] = structure_direction
        candles["bos_detected"]        = bos_detected
        candles["choch_detected"]      = choch_detected
        candles["structure_direction"] = candles["structure_direction"].ffill()
        return candles

    def _compute_fvg(self, candles: pd.DataFrame) -> pd.DataFrame:
        n       = len(candles)
        highs   = candles["bar_high"].values
        lows    = candles["bar_low"].values
        times   = candles["bar_time"].values
        records = []

        for i in range(1, n - 1):
            prev_high = highs[i - 1]
            prev_low  = lows[i - 1]
            next_high = highs[i + 1]
            next_low  = lows[i + 1]

            if prev_high < next_low:
                records.append({
                    "formed_at":  times[i],
                    "fvg_high":   round(float(next_low),  5),
                    "fvg_low":    round(float(prev_high), 5),
                    "fvg_side":   "bullish_fvg",
                    "fvg_filled": False,
                })
            elif prev_low > next_high:
                records.append({
                    "formed_at":  times[i],
                    "fvg_high":   round(float(prev_low),  5),
                    "fvg_low":    round(float(next_high), 5),
                    "fvg_side":   "bearish_fvg",
                    "fvg_filled": False,
                })

        if not records:
            return pd.DataFrame(columns=[
                "formed_at", "fvg_high", "fvg_low", "fvg_side", "fvg_filled",
            ])

        fvg_df = pd.DataFrame(records)
        fvg_df["formed_at"] = pd.to_datetime(fvg_df["formed_at"], utc=True)
        return fvg_df

    def _mark_filled_fvgs(
        self, fvg_df: pd.DataFrame, candles: pd.DataFrame
    ) -> pd.DataFrame:
        fvg_df = fvg_df.copy()
        for idx, row in fvg_df.iterrows():
            formed_at = row["formed_at"]
            future    = candles[candles["bar_time"] > formed_at]
            if future.empty:
                continue
            if row["fvg_side"] == "bullish_fvg":
                filled = (future["bar_low"] <= row["fvg_low"]).any()
            else:
                filled = (future["bar_high"] >= row["fvg_high"]).any()
            fvg_df.at[idx, "fvg_filled"] = filled
        return fvg_df

    def _identify_liquidity_levels(
        self, candles: pd.DataFrame, timeframe: str
    ) -> pd.DataFrame:
        records: list[dict] = []
        highs  = candles["bar_high"].values
        lows   = candles["bar_low"].values
        closes = candles["bar_close"].values
        times  = candles["bar_time"].values

        atr_col = f"atr_{ATR_PERIOD}"
        avg_atr = candles[atr_col].median() if atr_col in candles.columns else 0.0
        if avg_atr == 0 or np.isnan(avg_atr):
            avg_atr = (highs.max() - lows.min()) * 0.01

        n = len(candles)

        for i in range(BASE_SWING_WINDOW, n - BASE_SWING_WINDOW):
            local_atr = candles[atr_col].iloc[i] if atr_col in candles.columns else avg_atr
            if np.isnan(local_atr) or local_atr == 0:
                local_atr = avg_atr

            w = self._dynamic_swing_window(local_atr, avg_atr)
            if i < w or i >= n - w:
                continue

            window_highs = highs[i - w: i + w + 1]
            if highs[i] >= window_highs.max() - 1e-8:
                swing_size = highs[i] - lows[i]
                if swing_size >= local_atr * MIN_SWING_ATR_RATIO:
                    confirmed = self._confirm_swing_high(closes, lows, i, n)
                    records.append({
                        "bar_time":   times[i],
                        "liq_level":  round(float(highs[i]), 5),
                        "liq_type":   "swing_high",
                        "liq_side":   "buystops_above",
                        "timeframe":  timeframe,
                        "confirmed":  confirmed,
                        "swept":      False,
                        "swing_size": round(float(swing_size), 5),
                        "touch_count": 1,
                    })

            window_lows = lows[i - w: i + w + 1]
            if lows[i] <= window_lows.min() + 1e-8:
                swing_size = highs[i] - lows[i]
                if swing_size >= local_atr * MIN_SWING_ATR_RATIO:
                    confirmed = self._confirm_swing_low(closes, highs, i, n)
                    records.append({
                        "bar_time":   times[i],
                        "liq_level":  round(float(lows[i]), 5),
                        "liq_type":   "swing_low",
                        "liq_side":   "sellstops_below",
                        "timeframe":  timeframe,
                        "confirmed":  confirmed,
                        "swept":      False,
                        "swing_size": round(float(swing_size), 5),
                        "touch_count": 1,
                    })

        price_min = candles["bar_low"].min()
        price_max = candles["bar_high"].max()
        step      = self._adaptive_round_step(price_max)
        first_level = (int(price_min / step) + 1) * step

        for level in np.arange(first_level, price_max + step, step):
            level    = round(float(level), 2)
            bar_time = self._first_touch_time(candles, level, avg_atr)
            if bar_time is None:
                continue
            records.append({
                "bar_time":    bar_time,
                "liq_level":   level,
                "liq_type":    "round_number",
                "liq_side":    "buystops_above",
                "timeframe":   timeframe,
                "confirmed":   True,
                "swept":       False,
                "swing_size":  0.0,
                "touch_count": self._count_touches(candles, level, avg_atr),
            })

        if not records:
            return pd.DataFrame(columns=[
                "bar_time", "liq_level", "liq_type", "liq_side",
                "timeframe", "confirmed", "swept", "swing_size",
                "touch_count", "liq_score",
            ])

        liq_df = pd.DataFrame(records)
        liq_df["bar_time"] = pd.to_datetime(liq_df["bar_time"], utc=True)

        for idx, row in liq_df.iterrows():
            if row["liq_type"] != "round_number":
                liq_df.at[idx, "touch_count"] = self._count_touches(
                    candles, row["liq_level"], avg_atr
                )
        return liq_df

    @staticmethod
    def _confirm_swing_high(
        closes: np.ndarray, lows: np.ndarray, i: int, n: int
    ) -> bool:
        swing_low  = lows[i]
        look_ahead = min(i + 21, n)
        return any(closes[j] < swing_low for j in range(i + 1, look_ahead))

    @staticmethod
    def _confirm_swing_low(
        closes: np.ndarray, highs: np.ndarray, i: int, n: int
    ) -> bool:
        swing_high = highs[i]
        look_ahead = min(i + 21, n)
        return any(closes[j] > swing_high for j in range(i + 1, look_ahead))

    def _mark_swept_levels(
        self, liq_df: pd.DataFrame, candles: pd.DataFrame
    ) -> pd.DataFrame:
        liq_df = liq_df.copy()
        liq_df["swept_at"] = pd.NaT
        closes = candles["bar_close"].values
        times  = candles["bar_time"].values

        for idx, row in liq_df.iterrows():
            level     = row["liq_level"]
            formed_at = row["bar_time"]
            side      = row["liq_side"]

            future_mask   = candles["bar_time"] > formed_at
            future_closes = closes[future_mask]
            future_times  = times[future_mask]

            if side == "buystops_above":
                hit = future_closes > level * (1 + SWEEP_TOLERANCE)
            else:
                hit = future_closes < level * (1 - SWEEP_TOLERANCE)

            if hit.any():
                liq_df.at[idx, "swept"]    = True
                liq_df.at[idx, "swept_at"] = future_times[np.argmax(hit)]
        return liq_df

    def _score_confluence(self, liq_df: pd.DataFrame) -> pd.DataFrame:
        liq_df = liq_df.copy()
        levels = liq_df["liq_level"].values
        scores = np.zeros(len(liq_df), dtype=float)

        for i, row in liq_df.iterrows():
            if row.get("swept", False):
                scores[i] = -99
                continue
            level = row["liq_level"]
            nearby   = np.abs(levels - level) / (level + 1e-10) <= CONFLUENCE_TOLERANCE
            tf_count = liq_df[nearby]["timeframe"].nunique()
            scores[i] += tf_count - 1
            if row.get("confirmed", False):
                scores[i] += 1
            scores[i] += max(0, int(row.get("touch_count", 1)) - 1)
            if row["timeframe"] == "1d":
                scores[i] += 3
            elif row["timeframe"] == "4h":
                scores[i] += 2
            elif row["timeframe"] == "1h":
                scores[i] += 1
            if pd.notna(row["bar_time"]):
                hour = pd.Timestamp(row["bar_time"]).hour
                kz_start, kz_end = SESSIONS["killzone"]
                if kz_start <= hour < kz_end:
                    scores[i] += 2

        liq_df["liq_score"] = scores
        return liq_df

    def _merge_to_ticks(
        self,
        ticks_df: pd.DataFrame,
        candles:  pd.DataFrame,
        liq_df:   pd.DataFrame,
        fvg_df:   pd.DataFrame,
    ) -> pd.DataFrame:
        candles_sorted = candles.sort_values("bar_time")
        ticks_sorted   = ticks_df.sort_values("timestamp_utc")

        candle_cols = [
            "bar_time", "bar_open", "bar_high", "bar_low", "bar_close",
            f"rsi_{RSI_PERIOD}", f"atr_{ATR_PERIOD}",
            "structure_direction", "bos_detected", "choch_detected",
        ]
        available_cols = [c for c in candle_cols if c in candles_sorted.columns]

        enriched = pd.merge_asof(
            ticks_sorted,
            candles_sorted[available_cols],
            left_on="timestamp_utc",
            right_on="bar_time",
            direction="backward",
        )
        enriched = enriched.rename(columns={
            f"rsi_{RSI_PERIOD}": "rsi_14",
            f"atr_{ATR_PERIOD}": "atr_14",
            "fvg_filled": "fvg_filled_candle",
        })

        if not liq_df.empty:
            active_liq = liq_df[
                (~liq_df["swept"]) & (liq_df["liq_score"] >= 0)
            ].copy()
            if not active_liq.empty:
                enriched = self._attach_nearest_levels(enriched, active_liq)
            else:
                enriched = self._add_empty_liq_columns(enriched)
        else:
            enriched = self._add_empty_liq_columns(enriched)

        if not fvg_df.empty:
            active_fvg = fvg_df[~fvg_df["fvg_filled"]].copy()
            if not active_fvg.empty:
                enriched = self._attach_nearest_fvg(enriched, active_fvg, candles_sorted)
            else:
                enriched = self._add_empty_fvg_columns(enriched)
        else:
            enriched = self._add_empty_fvg_columns(enriched)

        enriched.drop(columns=["mid", "bar_time"], errors="ignore", inplace=True)
        return enriched

    def _attach_nearest_levels(
        self, enriched: pd.DataFrame, liq_df: pd.DataFrame
    ) -> pd.DataFrame:
        enriched  = enriched.copy()
        levels    = liq_df["liq_level"].values
        types     = liq_df["liq_type"].values
        sides     = liq_df["liq_side"].values
        tfs       = liq_df["timeframe"].values
        scores    = liq_df["liq_score"].values
        confirmed = liq_df["confirmed"].values
        swept     = liq_df["swept"].values

        nearest_level  = []
        nearest_type   = []
        nearest_side   = []
        nearest_tf     = []
        nearest_score  = []
        nearest_conf   = []
        nearest_swept_ = []
        dist_high      = []
        dist_low       = []

        for mid in enriched["mid"]:
            above_mask = levels > mid
            below_mask = levels < mid

            if above_mask.any():
                above_idx     = np.where(above_mask)[0]
                closest_above = above_idx[np.argmin(levels[above_idx] - mid)]
                dist_high.append(round(levels[closest_above] - mid, 5))
            else:
                closest_above = None
                dist_high.append(np.nan)

            if below_mask.any():
                below_idx     = np.where(below_mask)[0]
                closest_below = below_idx[np.argmin(mid - levels[below_idx])]
                dist_low.append(round(mid - levels[closest_below], 5))
            else:
                closest_below = None
                dist_low.append(np.nan)

            distances = np.abs(levels - mid)
            top_n_idx = np.argsort(distances)[:N_NEAREST_LEVELS]
            best_idx  = top_n_idx[np.argmax(scores[top_n_idx])]

            nearest_level.append(round(float(levels[best_idx]), 5))
            nearest_type.append(types[best_idx])
            nearest_side.append(sides[best_idx])
            nearest_tf.append(tfs[best_idx])
            nearest_score.append(float(scores[best_idx]))
            nearest_conf.append(bool(confirmed[best_idx]))
            nearest_swept_.append(bool(swept[best_idx]))

        enriched["liq_level"]            = nearest_level
        enriched["liq_type"]             = nearest_type
        enriched["liq_side"]             = nearest_side
        enriched["liq_tf"]               = nearest_tf
        enriched["liq_score"]            = nearest_score
        enriched["liq_confirmed"]        = nearest_conf
        enriched["liq_swept"]            = nearest_swept_
        enriched["dist_to_nearest_high"] = dist_high
        enriched["dist_to_nearest_low"]  = dist_low
        return enriched

    def _attach_nearest_fvg(
        self,
        enriched:  pd.DataFrame,
        fvg_df:    pd.DataFrame,
        candles:   pd.DataFrame,
    ) -> pd.DataFrame:
        enriched = enriched.copy()

        fvg_highs     = fvg_df["fvg_high"].values
        fvg_lows      = fvg_df["fvg_low"].values
        fvg_sides     = fvg_df["fvg_side"].values
        fvg_formed_at = pd.to_datetime(fvg_df["formed_at"].values, utc=True)

        bar_times = pd.to_datetime(candles["bar_time"].values, utc=True)

        out_fvg_high   = []
        out_fvg_low    = []
        out_fvg_side   = []
        out_fvg_filled = []
        out_fvg_age    = []

        for _, row in enriched.iterrows():
            mid       = row["mid"]
            tick_time = row["timestamp_utc"]
            direction = row.get("structure_direction")

            valid_mask = np.ones(len(fvg_df), dtype=bool)
            for k in range(len(fvg_df)):
                if direction == "bullish" and fvg_sides[k] != "bullish_fvg":
                    valid_mask[k] = False
                    continue
                if direction == "bearish" and fvg_sides[k] != "bearish_fvg":
                    valid_mask[k] = False
                    continue