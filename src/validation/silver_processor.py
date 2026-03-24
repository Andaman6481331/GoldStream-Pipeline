"""
Silver Layer — SilverProcessor
Converts raw ticks from MT5 (dict) and Dukascopy (Parquet partitions) into
the unified UnifiedTick schema.

Key change vs v1:
  - volume_usd field is now populated so both sources share a comparable
    notional scale before reaching the Gold layer.
      MT5        : lots  × 100_000  → USD notional
      Dukascopy  : raw   × 1_000_000 → USD notional  (Dukascopy unit = millions USD)
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

import pandas as pd
from pydantic import ValidationError

from src.validation.validators import DukascopyTick, UnifiedTick, RawTick

logger = logging.getLogger(__name__)

# ── Volume normalisation constants ────────────────────────────────────────────
MT5_LOT_TO_USD        = 100_000      # 1 standard lot ≈ $100,000 notional
DUKASCOPY_TO_USD      = 1_000_000    # Dukascopy volume unit = millions USD
MIN_NOTIONAL_USD      = 1_000        # ticks below $1,000 notional are quote noise


class SilverProcessor:
    """
    Unified Silver-layer processor.

    Sources:
        - MT5 real-time stream         →  process_mt5_tick(raw_dict)
        - Dukascopy Bronze Parquet     →  process_dukascopy_parquet(path)

    Both return / yield UnifiedTick instances ready for the Gold layer.
    Volume is normalised to USD notional so the Gold layer never needs to
    know which broker supplied a tick.
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
        UnifiedTick objects, skipping rows that fail validation or fall below
        the minimum notional threshold.

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

        if "timestamp_utc" not in df.columns:
            logger.error("[SilverProcessor] Missing 'timestamp_utc' column")
            return

        if df["timestamp_utc"].dt.tz is None:
            df["timestamp_utc"] = df["timestamp_utc"].dt.tz_localize("UTC")

        df = df.sort_values("timestamp_utc")
        ok, skipped_validation, skipped_notional = 0, 0, 0

        for _, row in df.iterrows():
            try:
                raw_tick = DukascopyTick(
                    timestamp_utc=row["timestamp_utc"].to_pydatetime(),
                    ask=float(row["ask"]),
                    bid=float(row["bid"]),
                    ask_volume=float(row["ask_volume"]),
                    bid_volume=float(row["bid_volume"]),
                    symbol=str(row.get("symbol", "XAUUSD")),
                )
            except ValidationError as exc:
                logger.debug(f"[SilverProcessor] Row skipped (validation): {exc}")
                skipped_validation += 1
                continue

            # Mean of bid/ask volume as representative tick volume
            raw_volume = (raw_tick.bid_volume + raw_tick.ask_volume) / 2.0
            volume_usd = raw_volume * DUKASCOPY_TO_USD

            if volume_usd < MIN_NOTIONAL_USD:
                skipped_notional += 1
                continue

            try:
                unified = UnifiedTick(
                    timestamp_utc=raw_tick.timestamp_utc,
                    symbol=raw_tick.symbol,
                    bid=raw_tick.bid,
                    ask=raw_tick.ask,
                    volume=raw_volume,
                    volume_usd=volume_usd,
                    source="dukascopy",
                )
                ok += 1
                yield unified
            except ValidationError as exc:
                logger.debug(f"[SilverProcessor] UnifiedTick failed: {exc}")
                skipped_validation += 1

        logger.info(
            f"[SilverProcessor] {path.name}: {ok} valid | "
            f"{skipped_validation} invalid | {skipped_notional} below notional"
        )

    # ── Batch helper ──────────────────────────────────────────────────────────

    def process_all_parquets(
        self, bronze_dir: str | Path, symbol: str = "XAUUSD"
    ) -> Generator[UnifiedTick, None, None]:
        """
        Walk the entire Bronze partition tree for a symbol and yield all
        UnifiedTick objects in chronological order.

        Layout: <bronze_dir>/<symbol>/year=*/month=*/ticks.parquet
        """
        bronze_dir = Path(bronze_dir)
        pattern = f"{symbol}/year=*/month=*/ticks.parquet"
        parquet_files = sorted(bronze_dir.glob(pattern))

        if not parquet_files:
            logger.warning(
                f"[SilverProcessor] No Parquet files found under {bronze_dir}/{symbol}"
            )
            return

        logger.info(
            f"[SilverProcessor] Processing {len(parquet_files)} Parquet partitions"
        )
        for pq_file in parquet_files:
            yield from self.process_dukascopy_parquet(pq_file)