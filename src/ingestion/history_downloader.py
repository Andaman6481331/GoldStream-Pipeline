"""
Bronze Layer — HistoryDownloader
Fetches Dukascopy .bi5 tick data directly from their CDN using aiohttp,
parses the LZMA-compressed binary payload with Python's struct module,
and saves the results as partitioned Parquet files (year/month).

No Node.js or dukascopy-python wrapper required.

CDN URL pattern:
  https://datafeed.dukascopy.com/datafeed/{SYMBOL}/{YYYY}/{MM:02d}/{DD:02d}/{HH:02d}h_ticks.bi5

Each .bi5 file contains rows of 5 big-endian values:
  uint32  delta_ms  — milliseconds since the start of the hour
  uint32  ask       — ask * 100000 (for 5-digit pairs)
  uint32  bid       — bid * 100000
  float32 ask_vol   — ask volume (lots)
  float32 bid_vol   — bid volume (lots)

Total: 20 bytes per tick row.
"""

import asyncio
import struct
import lzma
import logging
import random
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import aiohttp
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
DUKASCOPY_CDN = "https://datafeed.dukascopy.com/datafeed"
BI5_STRUCT_FMT = ">IIIff"       # big-endian: uint32, uint32, uint32, float32, float32
BI5_ROW_SIZE = struct.calcsize(BI5_STRUCT_FMT)   # 20 bytes
POINT_DIVISOR = 1000.0          # Adjusted for XAUUSD (2 or 3 decimal places)

# Dukascopy CDN often blocks non-browser agents
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"

# Parquet schema for Bronze layer
BRONZE_SCHEMA = pa.schema([
    pa.field("timestamp_utc", pa.timestamp("ms", tz="UTC")),
    pa.field("ask",           pa.float64()),
    pa.field("bid",           pa.float64()),
    pa.field("ask_volume",    pa.float64()),
    pa.field("bid_volume",    pa.float64()),
    pa.field("symbol",        pa.string()),
])


class HistoryDownloader:
    """
    Downloads Dukascopy historical tick data for a given symbol and date range,
    saves results as partitioned Parquet to the Bronze layer.

    Usage:
        downloader = HistoryDownloader(symbol="XAUUSD", output_dir="data/bronze")
        asyncio.run(downloader.download_range("2024-01-01", "2024-01-31"))
    """



    def __init__(
        self,
        symbol: str = "XAUUSD",
        output_dir: str = "data/bronze",
        max_concurrent: int = 4, # Reduced from 8 to prevent 503s
        request_timeout: int = 60,
    ):
        self.symbol = symbol.upper()
        self.output_dir = Path(output_dir)
        self.max_concurrent = max_concurrent
        self.timeout = aiohttp.ClientTimeout(total=request_timeout)
        self.headers = {"User-Agent": USER_AGENT}

    # ── Public API ────────────────────────────────────────────────────────────

    async def download_range(
        self,
        start: str,
        end: str,
    ) -> dict:
        """
        Download all hours between *start* and *end* (inclusive, YYYY-MM-DD format).
        Returns a summary dict with total ticks saved and files written.
        """
        start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt   = (
            datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            + timedelta(days=1)
        )

        # Build list of hourly URLs
        hours: list[datetime] = []
        cur = start_dt
        while cur < end_dt:
            hours.append(cur)
            cur += timedelta(hours=1)

        logger.info(
            f"[HistoryDownloader] Downloading {len(hours)} hourly files "
            f"for {self.symbol} ({start} → {end})"
        )

        semaphore = asyncio.Semaphore(self.max_concurrent)
        summary = {"ticks_saved": 0, "files_written": 0, "hours_skipped": 0}

        async with aiohttp.ClientSession(timeout=self.timeout, headers=self.headers) as session:
            tasks = [
                self._fetch_hour(session, semaphore, hour_dt, summary)
                for hour_dt in hours
            ]
            await asyncio.gather(*tasks)

        logger.info(
            f"[HistoryDownloader] Done. Ticks saved: {summary['ticks_saved']} | "
            f"Files written: {summary['files_written']} | "
            f"Hours skipped (empty/weekend): {summary['hours_skipped']}"
        )
        return summary

    # ── Internal helpers ─────────────────────────────────────────────────────

    async def _fetch_hour(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        hour_dt: datetime,
        summary: dict,
    ) -> None:
        url = self._build_url(hour_dt)
        max_retries = 3
        
        async with semaphore:
            for attempt in range(max_retries):
                try:
                    async with session.get(url) as resp:
                        if resp.status == 404:
                            summary["hours_skipped"] += 1
                            return
                        if resp.status == 503:
                            wait = (attempt + 1) * 5 + random.uniform(2, 5) # Increased wait + jitter
                            logger.warning(f"[HistoryDownloader] 503 (Throttled) for {hour_dt} - Retrying in {wait:.1f}s...")
                            await asyncio.sleep(wait)
                            continue
                            
                        resp.raise_for_status()
                        raw_bytes = await resp.read()
                        break # Success
                except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                    if attempt == max_retries - 1:
                        logger.error(f"[HistoryDownloader] Final failure for {url}: {exc}")
                        summary["hours_skipped"] += 1
                        return
                    wait = (attempt + 1) * 2
                    logger.warning(f"[HistoryDownloader] Error for {hour_dt}: {exc}. Retry {attempt+1}/{max_retries}")
                    await asyncio.sleep(wait)
            else:
                summary["hours_skipped"] += 1
                return

        df = self._parse_bi5(raw_bytes, hour_dt)
        if df is None or df.empty:
            summary["hours_skipped"] += 1
            return

        self._save_parquet(df, hour_dt)
        summary["ticks_saved"] += len(df)
        summary["files_written"] += 1

    def _build_url(self, hour_dt: datetime) -> str:
        # Dukascopy months are 0-indexed in the CDN path
        return (
            f"{DUKASCOPY_CDN}/{self.symbol}"
            f"/{hour_dt.year}"
            f"/{hour_dt.month - 1:02d}"
            f"/{hour_dt.day:02d}"
            f"/{hour_dt.hour:02d}h_ticks.bi5"
        )

    def _parse_bi5(self, data: bytes, hour_dt: datetime) -> Optional[pd.DataFrame]:
        """
        Decompress LZMA payload and unpack binary rows into a DataFrame.
        Returns None if the file is empty or cannot be parsed.
        """
        if not data:
            return None

        try:
            decompressed = lzma.decompress(data)
        except lzma.LZMAError as exc:
            logger.warning(f"[HistoryDownloader] LZMA decode error: {exc}")
            return None

        n_rows = len(decompressed) // BI5_ROW_SIZE
        if n_rows == 0:
            return None

        rows = []
        hour_epoch_ms = int(hour_dt.timestamp() * 1000)

        for i in range(n_rows):
            offset = i * BI5_ROW_SIZE
            chunk = decompressed[offset: offset + BI5_ROW_SIZE]
            if len(chunk) < BI5_ROW_SIZE:
                break
            delta_ms, ask_raw, bid_raw, ask_vol, bid_vol = struct.unpack(
                BI5_STRUCT_FMT, chunk
            )
            ts_ms = hour_epoch_ms + delta_ms
            ask   = ask_raw / POINT_DIVISOR
            bid   = bid_raw / POINT_DIVISOR
            rows.append((ts_ms, ask, bid, float(ask_vol), float(bid_vol)))

        if not rows:
            return None

        df = pd.DataFrame(rows, columns=["ts_ms", "ask", "bid", "ask_volume", "bid_volume"])
        df["timestamp_utc"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        df["symbol"] = self.symbol
        df.drop(columns=["ts_ms"], inplace=True)
        return df[["timestamp_utc", "ask", "bid", "ask_volume", "bid_volume", "symbol"]]

    def _save_parquet(self, df: pd.DataFrame, hour_dt: datetime) -> Path:
        """
        Write a single-hour DataFrame to a Hive-partitioned Parquet path:
          <output_dir>/<symbol>/year=<Y>/month=<M>/ticks.parquet

        Uses append mode so multiple hours in the same month accumulate cleanly.
        """
        partition_dir = (
            self.output_dir
            / self.symbol
            / f"year={hour_dt.year}"
            / f"month={hour_dt.month}"
        )
        partition_dir.mkdir(parents=True, exist_ok=True)

        out_path = partition_dir / "ticks.parquet"
        table = pa.Table.from_pandas(df, schema=BRONZE_SCHEMA, preserve_index=False)

        if out_path.exists():
            # Append to existing Parquet file for this month partition
            existing = pq.read_table(str(out_path))
            # Ensure the existing table has the exact same schema columns in order
            # (Removes any accidental partition/metadata columns like 'year'/'month')
            existing = existing.select(BRONZE_SCHEMA.names)
            combined = pa.concat_tables([existing, table])
            # Deduplicate by timestamp + symbol
            combined_df = combined.to_pandas().drop_duplicates(
                subset=["timestamp_utc", "symbol"]
            ).sort_values("timestamp_utc")
            table = pa.Table.from_pandas(combined_df, schema=BRONZE_SCHEMA, preserve_index=False)

        pq.write_table(table, str(out_path), compression="snappy")
        logger.debug(f"[HistoryDownloader] Saved {len(df)} rows → {out_path}")
        return out_path
