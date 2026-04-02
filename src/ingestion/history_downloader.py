"""
Bronze Layer — HistoryDownloader
Fetches Dukascopy .bi5 tick data directly from their CDN using aiohttp,
parses the LZMA-compressed binary payload with Python's struct module,
and saves the results as partitioned Parquet files (year/month/DD_HH).

CDN URL pattern:
  https://datafeed.dukascopy.com/datafeed/{SYMBOL}/{YYYY}/{MM:02d}/{DD:02d}/{HH:02d}h_ticks.bi5

Each .bi5 file contains rows of 5 big-endian values:
  uint32  delta_ms  — milliseconds since the start of the hour
  uint32  ask       — ask × 100_000  (XAUUSD and all non-JPY instruments)
  uint32  bid       — bid × 100_000
  float32 ask_vol   — ask volume (lots, multiply × 1_000_000 for real units)
  float32 bid_vol   — bid volume

Total: 20 bytes per tick row.

503 mitigation strategy:
  - max_concurrent default reduced to 2 (Dukascopy CDN throttles at ≥3)
  - Mandatory inter-request delay (REQUEST_GAP_S) between all requests
  - Full exponential backoff with wide jitter to prevent thundering herd
  - max_retries increased to 8 — sustained 503 storms need more patience
  - Explicit TCPConnector limit matches max_concurrent — no silent overrun
  - Session-level connector limit prevents burst spikes during task startup
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

# ── Constants ─────────────────────────────────────────────────────────────────

DUKASCOPY_CDN  = "https://datafeed.dukascopy.com/datafeed"
BI5_STRUCT_FMT = ">IIIff"
BI5_ROW_SIZE   = struct.calcsize(BI5_STRUCT_FMT)   # 20 bytes

# Dukascopy encodes raw tick integers differently by instrument class:
#   FX non-JPY (e.g. EURUSD):  integer = price × 100_000  (5 decimal places)
#   Metals/CFDs (e.g. XAUUSD): integer = price × 1_000    (3 decimal places)
#   JPY pairs (e.g. USDJPY):   integer = price × 1_000    (3 decimal places)
# Using the wrong divisor produces prices ~100× too small (e.g. 45 instead of 4,500).
FX_POINT_DIVISOR    = 100_000.0
METAL_POINT_DIVISOR = 1_000.0
METAL_SYMBOLS = {"XAUUSD", "XAGUSD", "XAUEUR", "XAGEUR", "XPTUSD", "XPDUSD"}

# Dukascopy CDN requires a browser-like UA or returns 403/503 immediately.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# Rate-limiting tuning — adjust these if you still hit 503s
MAX_CONCURRENT  = 2      # safe concurrent requests to Dukascopy CDN
REQUEST_GAP_S   = 1.5    # minimum seconds between any two requests (even after semaphore release)
MAX_RETRIES     = 8      # Dukascopy 503 storms can last several minutes

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

    Partition layout:
        <output_dir>/<SYMBOL>/year=<YYYY>/month=<MM:02d>/<DD:02d>_<HH:02d>.parquet

    Each hour is its own file.  Call merge_month() to consolidate a full
    month partition into a single ticks.parquet for downstream consumers.

    Usage:
        downloader = HistoryDownloader(symbol="XAUUSD", output_dir="data/bronze")
        asyncio.run(downloader.download_range("2024-01-01", "2024-01-31"))
    """

    def __init__(
        self,
        symbol:          str  = "XAUUSD",
        output_dir:      str  = "data/bronze",
        max_concurrent:  int  = MAX_CONCURRENT,
        max_retries:     int  = MAX_RETRIES,
        request_timeout: int  = 60,
        request_gap_s:   float = REQUEST_GAP_S,
    ):
        self.symbol         = symbol.upper()
        self.output_dir     = Path(output_dir)
        self.max_concurrent = max_concurrent
        self.max_retries    = max_retries
        self.request_gap_s  = request_gap_s
        self.timeout        = aiohttp.ClientTimeout(total=request_timeout)
        self.headers        = {"User-Agent": USER_AGENT}

        # _last_request_at guards the inter-request gap across all coroutines.
        self._last_request_at: float = 0.0

    # ── Public API ─────────────────────────────────────────────────────────────

    async def download_range(self, start: str, end: str) -> dict:
        """
        Download all trading hours between *start* and *end* (inclusive,
        YYYY-MM-DD format). Weekend hours and already-downloaded hours are
        skipped without any network request.

        Returns a summary dict:
            ticks_saved, files_written, hours_skipped, hours_resumed
        """
        start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt   = (
            datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            + timedelta(days=1)
        )

        hours: list[datetime] = []
        cur = start_dt
        while cur < end_dt:
            if cur.weekday() < 5:   # 0=Mon … 4=Fri
                hours.append(cur)
            cur += timedelta(hours=1)

        logger.info(
            f"[HistoryDownloader] {self.symbol} {start} → {end} | "
            f"{len(hours)} trading hours queued (weekends pre-filtered)"
        )

        semaphore = asyncio.Semaphore(self.max_concurrent)
        lock      = asyncio.Lock()
        summary   = {
            "ticks_saved":   0,
            "files_written": 0,
            "hours_skipped": 0,
            "hours_resumed": 0,
        }

        # Explicit connector limit matches max_concurrent — prevents the
        # aiohttp default pool from silently opening more connections than
        # intended during the burst at task startup.
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)

        async with aiohttp.ClientSession(
            timeout=self.timeout,
            headers=self.headers,
            connector=connector,
        ) as session:
            tasks = [
                self._fetch_hour(session, semaphore, lock, hour_dt, summary)
                for hour_dt in hours
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, Exception):
                logger.error(f"[HistoryDownloader] Unhandled task exception: {r}")

        logger.info(
            f"[HistoryDownloader] Done. "
            f"ticks_saved={summary['ticks_saved']:,} | "
            f"files_written={summary['files_written']} | "
            f"hours_skipped={summary['hours_skipped']} | "
            f"hours_resumed={summary['hours_resumed']}"
        )
        return summary

    def merge_month(self, year: int, month: int) -> Optional[Path]:
        """
        Consolidate all per-hour Parquet files for a given month into a single
        ticks.parquet. Deduplicates by (timestamp_utc, symbol) and sorts ascending.
        Returns the output path, or None if no per-hour files were found.
        """
        partition_dir = (
            self.output_dir / self.symbol
            / f"year={year}"
            / f"month={month:02d}"
        )
        hour_files = sorted(partition_dir.glob("[0-9][0-9]_[0-9][0-9].parquet"))
        if not hour_files:
            logger.warning(f"[HistoryDownloader] merge_month: no hour files in {partition_dir}")
            return None

        tables   = [pq.read_table(str(f)) for f in hour_files]
        combined = pa.concat_tables(tables)
        combined_df = (
            combined.to_pandas()
            .drop_duplicates(subset=["timestamp_utc", "symbol"])
            .sort_values("timestamp_utc")
        )
        out_table = pa.Table.from_pandas(
            combined_df, schema=BRONZE_SCHEMA, preserve_index=False
        )
        out_path = partition_dir / "ticks.parquet"
        pq.write_table(out_table, str(out_path), compression="snappy")

        logger.info(
            f"[HistoryDownloader] Merged {len(hour_files)} hour files → "
            f"{len(combined_df):,} rows → {out_path}"
        )
        return out_path

    # ── Internal helpers ───────────────────────────────────────────────────────

    async def _throttle(self) -> None:
        """
        Enforce a minimum gap of request_gap_s between requests regardless of
        concurrency. This is the primary defence against Dukascopy rate-limiting.

        Without this, all coroutines that clear the semaphore simultaneously
        fire at the CDN in the same millisecond — the semaphore controls
        parallelism but not request rate.
        """
        async with asyncio.Lock():
            now  = asyncio.get_event_loop().time()
            wait = self._last_request_at + self.request_gap_s - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request_at = asyncio.get_event_loop().time()

    async def _fetch_hour(
        self,
        session:   aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        lock:      asyncio.Lock,
        hour_dt:   datetime,
        summary:   dict,
    ) -> None:
        # Resume: skip HTTP entirely if already on disk.
        out_path = self._hour_parquet_path(hour_dt)
        if out_path.exists():
            async with lock:
                summary["hours_resumed"] += 1
            return

        url       = self._build_url(hour_dt)
        raw_bytes: Optional[bytes] = None

        async with semaphore:
            # Throttle INSIDE the semaphore so the rate limit applies per-slot,
            # not per-task. Tasks waiting on the semaphore do not consume quota.
            await self._throttle()

            for attempt in range(self.max_retries):
                try:
                    async with session.get(url) as resp:

                        if resp.status == 404:
                            # No data for this hour — holiday / off-hours gap
                            async with lock:
                                summary["hours_skipped"] += 1
                            return

                        if resp.status in (429, 502, 503):
                            # Exponential backoff with wide jitter.
                            # Wide jitter (0 – base_wait) scatters retries from
                            # multiple coroutines so they don't reconverge.
                            base_wait = min(2 ** attempt * 5, 120)  # caps at 120s
                            wait      = base_wait + random.uniform(0, base_wait)
                            logger.warning(
                                f"[HistoryDownloader] HTTP {resp.status} for {hour_dt} "
                                f"— retry {attempt + 1}/{self.max_retries} "
                                f"in {wait:.1f}s"
                            )
                            await asyncio.sleep(wait)
                            # Re-throttle after sleep so the next attempt also
                            # respects the inter-request gap.
                            await self._throttle()
                            continue

                        resp.raise_for_status()
                        raw_bytes = await resp.read()
                        break   # success

                except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                    if attempt == self.max_retries - 1:
                        logger.error(
                            f"[HistoryDownloader] Final failure for {url}: {exc}"
                        )
                        async with lock:
                            summary["hours_skipped"] += 1
                        return
                    # Linear backoff + jitter for network errors (they resolve faster)
                    wait = (attempt + 1) * 4 + random.uniform(1, 5)
                    logger.warning(
                        f"[HistoryDownloader] {type(exc).__name__} for {hour_dt} "
                        f"— retry {attempt + 1}/{self.max_retries} in {wait:.1f}s"
                    )
                    await asyncio.sleep(wait)
                    await self._throttle()
            else:
                # All retries exhausted on 503/502/429
                logger.error(
                    f"[HistoryDownloader] Gave up on {hour_dt} after "
                    f"{self.max_retries} attempts"
                )
                async with lock:
                    summary["hours_skipped"] += 1
                return

        if raw_bytes is None:
            async with lock:
                summary["hours_skipped"] += 1
            return

        df = self._parse_bi5(raw_bytes, hour_dt)
        if df is None or df.empty:
            async with lock:
                summary["hours_skipped"] += 1
            return

        self._save_hour_parquet(df, hour_dt)
        async with lock:
            summary["ticks_saved"]   += len(df)
            summary["files_written"] += 1

    def _build_url(self, hour_dt: datetime) -> str:
        # Dukascopy CDN months are 0-indexed (Jan = 00)
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
        Returns None if the payload is empty or cannot be parsed.
        """
        if not data:
            return None

        try:
            decompressed = lzma.decompress(data)
        except lzma.LZMAError as exc:
            logger.warning(f"[HistoryDownloader] LZMA decode error for {hour_dt}: {exc}")
            return None

        n_rows = len(decompressed) // BI5_ROW_SIZE
        if n_rows == 0:
            return None

        hour_epoch_ms = int(hour_dt.timestamp() * 1000)
        rows = []

        # Select the correct divisor for this symbol:
        #   metals/CFDs use 1,000  (3 dp)  — e.g. XAUUSD raw 4500000 → 4500.000
        #   FX non-JPY  use 100,000 (5 dp) — e.g. EURUSD raw 108345  → 1.08345
        divisor = METAL_POINT_DIVISOR if self.symbol in METAL_SYMBOLS else FX_POINT_DIVISOR

        for i in range(n_rows):
            offset = i * BI5_ROW_SIZE
            chunk  = decompressed[offset: offset + BI5_ROW_SIZE]
            if len(chunk) < BI5_ROW_SIZE:
                break
            delta_ms, ask_raw, bid_raw, ask_vol, bid_vol = struct.unpack(
                BI5_STRUCT_FMT, chunk
            )
            ts_ms = hour_epoch_ms + delta_ms
            ask   = ask_raw / divisor
            bid   = bid_raw / divisor
            # Dukascopy volume is in millions of units — multiply to get real lots
            rows.append((
                ts_ms,
                ask,
                bid,
                float(ask_vol) * 1_000_000,
                float(bid_vol) * 1_000_000,
            ))

        if not rows:
            return None

        df = pd.DataFrame(
            rows, columns=["ts_ms", "ask", "bid", "ask_volume", "bid_volume"]
        )
        df["timestamp_utc"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        df["symbol"]        = self.symbol
        df.drop(columns=["ts_ms"], inplace=True)
        return df[["timestamp_utc", "ask", "bid", "ask_volume", "bid_volume", "symbol"]]

    def _hour_parquet_path(self, hour_dt: datetime) -> Path:
        """
        Canonical per-hour Parquet path. Day is included in the filename
        to prevent same-hour collisions across different days.
        """
        return (
            self.output_dir
            / self.symbol
            / f"year={hour_dt.year}"
            / f"month={hour_dt.month:02d}"
            / f"{hour_dt.day:02d}_{hour_dt.hour:02d}.parquet"
        )

    def _save_hour_parquet(self, df: pd.DataFrame, hour_dt: datetime) -> Path:
        """Write a single hour as its own Parquet file."""
        out_path = self._hour_parquet_path(hour_dt)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        table = pa.Table.from_pandas(df, schema=BRONZE_SCHEMA, preserve_index=False)
        pq.write_table(table, str(out_path), compression="snappy")

        logger.debug(f"[HistoryDownloader] Saved {len(df)} rows → {out_path}")
        return out_path