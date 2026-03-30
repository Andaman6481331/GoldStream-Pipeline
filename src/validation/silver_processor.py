"""
Silver Layer — SilverProcessor
Converts raw ticks from MT5 (dict) and Dukascopy (Parquet partitions) into
the unified UnifiedTick schema.

Volume normalisation:
  MT5        : lots × 100_000                      → USD notional
  Dukascopy  : volume_oz × mid_price (bid+ask)/2   → USD notional

  IMPORTANT — Dukascopy XAUUSD volume is in TROY OUNCES, not FX millions.
  The old DUKASCOPY_TO_USD = 1_000_000 fixed multiplier was correct only for
  FX pairs. For XAUUSD the correct conversion is:
      volume_usd = volume_oz × mid_price
  where mid_price = (bid + ask) / 2 at that tick.
  There is no valid fixed multiplier — the spot price must be used per tick.

Fix log vs previous version:
    [HIGH] Zero-spread warning moved BEFORE DukascopyTick construction.
           Previous code checked raw_tick.ask == raw_tick.bid after the
           validator already rejected zero-spread rows with ValidationError,
           making the warning dead code. The check now runs on the raw floats
           from the DataFrame before the Pydantic model is constructed.
    [HIGH] process_dukascopy_parquet rewritten with vectorised bulk filtering.
           iterrows() on 2–5M tick monthly Parquet files was O(n) Python loop
           overhead per row. All structural filters (price > 0, ask >= bid,
           notional floor) are now applied as pandas operations. UnifiedTick
           objects are only constructed for rows that pass all filters.
    [MINOR] Timezone guard added to process_all_parquets: tz-naive Timestamps
            from pd.to_datetime(string) are localised to UTC before any
            date arithmetic to prevent mixed-tz comparison errors downstream.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import pandas as pd
from pydantic import ValidationError

from src.validation.validators import UnifiedTick, RawTick

logger = logging.getLogger(__name__)

# ── Volume normalisation constants ────────────────────────────────────────────
MT5_LOT_TO_USD   = 100_000   # 1 standard lot ≈ $100,000 notional
# NOTE: No fixed Dukascopy multiplier — XAUUSD volume is in troy ounces.
#       USD notional is computed per-tick as: volume_oz × mid_price.
#       See _dukascopy_volume_usd().

# Minimum USD notional to accept a tick. 1_000 is appropriate now that the
# volume conversion is correct. Re-evaluate via backtesting if needed.
MIN_NOTIONAL_USD = 1_000


def _dukascopy_volume_usd(volume_oz: float, bid: float, ask: float) -> float:
    """
    Convert a Dukascopy XAUUSD tick volume (troy ounces) to USD notional.

    Dukascopy reports tick volume in troy ounces for XAUUSD (not FX millions).
    We use the mid-price at that tick as the spot price to convert:
        volume_usd = volume_oz × (bid + ask) / 2

    Args:
        volume_oz: Mean of ask_volume and bid_volume from the Dukascopy row.
        bid:       Bid price at that tick.
        ask:       Ask price at that tick.

    Returns:
        USD notional value of the tick.
    """
    mid = (bid + ask) / 2.0
    return volume_oz * mid


def _ensure_utc(dt) -> datetime:
    """
    Guarantee a UTC-aware datetime regardless of whether the input is a
    naive datetime, a tz-naive pd.Timestamp, or already UTC-aware.
    """
    if isinstance(dt, pd.Timestamp):
        if dt.tz is None:
            dt = dt.tz_localize("UTC")
        return dt.to_pydatetime()
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    # Fallback: parse as string
    ts = pd.to_datetime(dt)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return ts.to_pydatetime()


class SilverProcessor:
    """
    Unified Silver-layer processor.

    Sources:
        - MT5 real-time stream         →  process_mt5_tick(raw_dict)
        - Dukascopy Bronze Parquet     →  process_dukascopy_parquet(path)

    Both return / yield UnifiedTick instances ready for the Gold layer.
    Volume is normalised to USD notional so the Gold layer never needs to
    know which broker supplied a tick.

    This layer does NOT compute any derived features (EMA, RSI, ATR, FVG,
    swing points, etc.). All feature engineering belongs in the Gold layer.
    """

    # ── MT5 path ──────────────────────────────────────────────────────────────

    def process_mt5_tick(self, raw: dict) -> UnifiedTick | None:
        """
        Adapt an MT5 raw tick dict (from mt5_client.fetch_tick) to UnifiedTick.
        Expected keys: symbol, bid, ask, last, volume, time_msc
        """
        try:
            validated = RawTick(**raw)
        except ValidationError as exc:
            logger.warning(f"[SilverProcessor] Invalid MT5 tick: {exc}")
            return None

        ts = datetime.fromtimestamp(validated.time_msc / 1000.0, tz=timezone.utc)
        volume_usd = validated.volume * MT5_LOT_TO_USD

        if volume_usd < MIN_NOTIONAL_USD:
            logger.debug(
                f"[SilverProcessor] MT5 tick below min notional "
                f"(${volume_usd:.0f}), skipping"
            )
            return None

        try:
            return UnifiedTick(
                timestamp_utc=ts,
                symbol=validated.symbol,
                bid=validated.bid,
                ask=validated.ask,
                volume=validated.volume,
                volume_usd=volume_usd,
                source="mt5",
            )
        except ValidationError as exc:
            logger.warning(f"[SilverProcessor] MT5 → UnifiedTick failed: {exc}")
            return None

    # ── Dukascopy path ────────────────────────────────────────────────────────

    def process_dukascopy_parquet(
        self, path: str | Path
    ) -> Generator[UnifiedTick, None, None]:
        """
        Read a Bronze Parquet file (one month partition) and yield validated
        UnifiedTick objects, skipping rows that fail structural checks or fall
        below the minimum notional threshold.

        Filtering is vectorised — no row-by-row Python loop.  UnifiedTick
        objects are only constructed for rows that pass all filters.

        Args:
            path: Path to a ticks.parquet file written by HistoryDownloader.

        Yields:
            UnifiedTick instances in ascending timestamp order.
        """
        path = Path(path)
        if not path.exists():
            logger.error(f"[SilverProcessor] Parquet not found: {path}")
            return

        try:
            df = pd.read_parquet(str(path))
        except Exception as exc:
            logger.error(f"[SilverProcessor] Failed to read Parquet: {exc}")
            return

        required = {"timestamp_utc", "ask", "bid", "ask_volume", "bid_volume"}
        missing  = required - set(df.columns)
        if missing:
            logger.error(f"[SilverProcessor] Missing columns: {missing}")
            return

        # ── Timezone normalisation ─────────────────────────────────────────
        if df["timestamp_utc"].dt.tz is None:
            df["timestamp_utc"] = df["timestamp_utc"].dt.tz_localize("UTC")

        df = df.sort_values("timestamp_utc").reset_index(drop=True)

        # Fill missing symbol column (older Bronze files may omit it)
        if "symbol" not in df.columns:
            df["symbol"] = "XAUUSD"
        else:
            df["symbol"] = df["symbol"].fillna("XAUUSD").astype(str)

        n_raw = len(df)

        # ── Vectorised structural filters ──────────────────────────────────

        # 1. Cast to float — coerce bad values to NaN so they can be dropped
        for col in ("ask", "bid", "ask_volume", "bid_volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # 2. Drop rows with any NaN in critical columns
        df = df.dropna(subset=["ask", "bid", "ask_volume", "bid_volume", "timestamp_utc"])

        # 3. Price must be positive
        df = df[(df["ask"] > 0) & (df["bid"] > 0)]

        # 4. Inverted price (ask < bid) is a hard reject.
        #    Zero-spread (ask == bid) is allowed — log count at INFO.
        inverted    = df["ask"] < df["bid"]
        zero_spread = df["ask"] == df["bid"]

        n_inverted    = int(inverted.sum())
        n_zero_spread = int(zero_spread.sum())

        if n_inverted:
            logger.warning(
                f"[SilverProcessor] {path.name}: {n_inverted} inverted-price "
                f"rows (ask < bid) dropped"
            )
        if n_zero_spread:
            logger.info(
                f"[SilverProcessor] {path.name}: {n_zero_spread} zero-spread "
                f"ticks (ask == bid) accepted — typical for off-hours data"
            )

        df = df[~inverted]   # drop inverted; keep zero-spread

        # 5. Compute volume and notional vectorised
        df["volume_oz"]  = (df["ask_volume"] + df["bid_volume"]) / 2.0
        df["volume_usd"] = df["volume_oz"] * ((df["bid"] + df["ask"]) / 2.0)

        # 6. Notional floor
        df_valid = df[df["volume_usd"] >= MIN_NOTIONAL_USD].copy()

        n_skipped_validation = n_raw - len(df) - n_inverted
        n_skipped_notional   = len(df) - len(df_valid)
        ok = 0

        # ── Construct UnifiedTick objects from clean rows ──────────────────
        # Pydantic construction is kept for downstream type-safety and to
        # catch any edge cases that slip past the vectorised filters.
        for row in df_valid.itertuples(index=False):
            try:
                unified = UnifiedTick(
                    timestamp_utc=row.timestamp_utc.to_pydatetime()
                        if hasattr(row.timestamp_utc, "to_pydatetime")
                        else row.timestamp_utc,
                    symbol=row.symbol,
                    bid=float(row.bid),
                    ask=float(row.ask),
                    volume=float(row.volume_oz),
                    volume_usd=float(row.volume_usd),
                    source="dukascopy",
                )
                ok += 1
                yield unified
            except ValidationError as exc:
                # Should be rare after vectorised pre-filtering
                logger.debug(f"[SilverProcessor] UnifiedTick construction failed: {exc}")
                n_skipped_validation += 1

        logger.info(
            f"[SilverProcessor] {path.name}: {ok:,} valid | "
            f"{n_skipped_validation} invalid | "
            f"{n_skipped_notional} below notional (${MIN_NOTIONAL_USD:,})"
        )

    # ── Batch helper ──────────────────────────────────────────────────────────

    def process_all_parquets(
        self,
        bronze_dir:  str | Path,
        symbol:      str                = "XAUUSD",
        start_date:  Optional[datetime] = None,
        end_date:    Optional[datetime] = None,
    ) -> Generator[UnifiedTick, None, None]:
        """
        Walk the entire Bronze partition tree for a symbol and yield all
        UnifiedTick objects in chronological order.

        Layout: <bronze_dir>/<symbol>/year=*/month=*/ticks.parquet
        """
        bronze_dir = Path(bronze_dir)
        parquet_files: list[Path] = []

        if start_date and end_date:
            # FIX: guarantee UTC-aware datetimes before any date arithmetic.
            # pd.to_datetime(string) produces tz-naive Timestamps; comparisons
            # with tz-aware data later in the pipeline would raise TypeError.
            start_date = _ensure_utc(start_date)
            end_date   = _ensure_utc(end_date)

            # pd.date_range with freq="MS" already starts at month=1 of the
            # start month — no .union() needed (was adding a duplicate).
            months = pd.date_range(
                start=pd.Timestamp(start_date).replace(day=1),
                end=pd.Timestamp(end_date),
                freq="MS",
            )

            for m in months:
                # Try zero-padded month first (canonical from HistoryDownloader
                # v2+), then unpadded for backward compatibility with older data.
                for month_str in (f"{m.month:02d}", str(m.month)):
                    tgt = (
                        bronze_dir
                        / symbol
                        / f"year={m.year}"
                        / f"month={month_str}"
                        / "ticks.parquet"
                    )
                    if tgt.exists():
                        parquet_files.append(tgt)
                        break   # found this month — don't add both variants

            # Resolve to canonical absolute paths before deduplication so that
            # a zero-padded and unpadded path pointing to the same inode collapse.
            parquet_files = sorted({p.resolve() for p in parquet_files})

        else:
            pattern = f"{symbol}/year=*/month=*/ticks.parquet"
            parquet_files = sorted(bronze_dir.glob(pattern))

        if not parquet_files:
            logger.warning(
                f"[SilverProcessor] No Parquet files found under "
                f"{bronze_dir}/{symbol}"
            )
            return

        logger.info(
            f"[SilverProcessor] Processing {len(parquet_files)} Parquet partitions"
        )
        for pq_file in parquet_files:
            yield from self.process_dukascopy_parquet(pq_file)