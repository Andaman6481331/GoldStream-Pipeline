"""
Validators — Pydantic schemas for Bronze → Silver → Gold pipeline.

Hierarchy:
    RawTick         — MT5 raw feed (Bronze)
    DukascopyTick   — Dukascopy .bi5 row (Bronze)
    UnifiedTick     — Normalised Silver contract (both sources → Gold)

Fix log vs original:
    [HIGH] DukascopyTick.ask_must_be_above_bid: changed ask <= bid → ask < bid.
           Dukascopy off-hours data legitimately contains zero-spread ticks
           (ask == bid). The strict <= was silently dropping valid ticks.
           MT5 RawTick intentionally keeps <= — a zero-spread live tick
           is a real data error for a streaming broker feed.
    [HIGH] UnifiedTick.ask_above_bid: same change, ask <= bid → ask < bid,
           so zero-spread ticks that passed DukascopyTick validation are not
           then rejected when constructing the unified contract.
    [MINOR] UnifiedTick docstring corrected: removed stale volume_usd
            description that referenced a fixed multiplier for FX pairs;
            now reflects the per-tick oz × mid_price conversion for XAUUSD.
    [MINOR] Removed v1 LiquidityLevel Pydantic class. It conflicted with the
            LiquidityLevel dataclass in feature_engineer.py (same name,
            different schema). The Gold-layer dataclass is the authoritative
            definition. Any imports of LiquidityLevel from this module must
            be updated to import from feature_engineer instead.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, field_validator, model_validator


# ── Bronze: MT5 ───────────────────────────────────────────────────────────────

class RawTick(BaseModel):
    """
    Validated representation of a single MT5 live tick.

    ask > bid is strictly required — a zero-spread tick from a live MT5
    broker feed is a genuine data error and should be rejected.
    """
    symbol:   str
    bid:      float
    ask:      float
    last:     float
    volume:   float
    time_msc: int

    @field_validator("bid", "ask")
    @classmethod
    def price_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Price must be greater than 0, got {v}")
        return v

    @field_validator("symbol")
    @classmethod
    def symbol_must_not_be_empty(cls, v: str) -> str:
        if not v or v.strip() == "":
            raise ValueError("Symbol cannot be empty")
        return v.strip().upper()

    @field_validator("time_msc")
    @classmethod
    def timestamp_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Timestamp must be positive")
        return v

    @model_validator(mode="after")
    def ask_must_be_above_bid(self) -> "RawTick":
        # Intentionally strict (<=): zero-spread is invalid for live MT5 feeds.
        if self.ask <= self.bid:
            raise ValueError(
                f"ask ({self.ask}) must be greater than bid ({self.bid})"
            )
        return self


# ── Bronze: Dukascopy ─────────────────────────────────────────────────────────

class DukascopyTick(BaseModel):
    """
    Validated representation of a single Dukascopy .bi5 tick row.

    Zero-spread ticks (ask == bid) are permitted — Dukascopy historical data
    legitimately contains zero-spread rows during off-hours and low-liquidity
    periods. The silver processor logs these at WARNING so they are visible
    without being rejected.
    """
    timestamp_utc: datetime
    ask:           float
    bid:           float
    ask_volume:    float
    bid_volume:    float
    symbol:        str = "XAUUSD"

    @field_validator("ask", "bid")
    @classmethod
    def price_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Price must be greater than 0, got {v}")
        return round(v, 5)

    @field_validator("ask_volume", "bid_volume")
    @classmethod
    def volume_must_be_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"Volume cannot be negative, got {v}")
        return v

    @model_validator(mode="after")
    def ask_must_not_be_below_bid(self) -> "DukascopyTick":
        # FIX: changed <= to < — zero-spread (ask == bid) is valid historical data.
        # Inverted ask (ask < bid) is still a hard reject.
        if self.ask < self.bid:
            raise ValueError(
                f"ask ({self.ask}) is below bid ({self.bid}) — inverted price rejected"
            )
        return self


# ── Silver: Unified contract ──────────────────────────────────────────────────

class UnifiedTick(BaseModel):
    """
    Common schema that BOTH MT5 and Dukascopy ticks normalise into.
    This is the input contract for the Gold layer.

    volume_usd: normalised notional in USD so both sources are comparable.
        MT5:        volume (lots) × 100_000
        Dukascopy:  volume (troy oz) × mid_price at that tick
                    There is no valid fixed multiplier for XAUUSD — the spot
                    price must be used per tick.
    """
    timestamp_utc: datetime
    symbol:        str
    bid:           float
    ask:           float
    volume:        float        # raw source volume (kept for traceability)
    volume_usd:    float        # normalised notional USD
    source:        Literal["mt5", "dukascopy"]

    @field_validator("bid", "ask")
    @classmethod
    def price_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Price must be > 0, got {v}")
        return round(v, 5)

    @model_validator(mode="after")
    def ask_above_bid(self) -> "UnifiedTick":
        # FIX: changed <= to < — zero-spread ticks from Dukascopy are valid
        # and must not be rejected here after passing DukascopyTick validation.
        if self.ask < self.bid:
            raise ValueError(
                f"ask ({self.ask}) is below bid ({self.bid}) — inverted price rejected"
            )
        return self

    @property
    def mid(self) -> float:
        return round((self.bid + self.ask) / 2.0, 5)

    @property
    def spread(self) -> float:
        return round(self.ask - self.bid, 5)