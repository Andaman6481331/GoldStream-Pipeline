"""
Validators — Pydantic schemas for Bronze → Silver → Gold pipeline.

Hierarchy:
    RawTick         — MT5 raw feed (Bronze)
    DukascopyTick   — Dukascopy .bi5 row (Bronze)
    UnifiedTick     — Normalised Silver contract (both sources → Gold)
    LiquidityLevel  — Enriched liquidity record (Gold output)
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, field_validator, model_validator


# ── Bronze: MT5 ───────────────────────────────────────────────────────────────

class RawTick(BaseModel):
    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
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
        if self.ask <= self.bid:
            raise ValueError(
                f"ask ({self.ask}) must be greater than bid ({self.bid})"
            )
        return self


# ── Bronze: Dukascopy ─────────────────────────────────────────────────────────

class DukascopyTick(BaseModel):
    """Validated representation of a single Dukascopy .bi5 tick row."""

    timestamp_utc: datetime
    ask: float
    bid: float
    ask_volume: float
    bid_volume: float
    symbol: str = "XAUUSD"

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
    def ask_must_be_above_bid(self) -> "DukascopyTick":
        if self.ask <= self.bid:
            raise ValueError(
                f"ask ({self.ask}) must be greater than bid ({self.bid})"
            )
        return self


# ── Silver: Unified contract ──────────────────────────────────────────────────

class UnifiedTick(BaseModel):
    """
    Common schema that BOTH MT5 and Dukascopy ticks normalise into.
    This is the input contract for the Gold layer.

    volume_usd: normalised notional in USD so both sources are comparable.
        MT5:        volume (lots) × 100_000
        Dukascopy:  volume (millions USD) × 1_000_000
    """

    timestamp_utc: datetime
    symbol: str
    bid: float
    ask: float
    volume: float               # raw source volume (kept for traceability)
    volume_usd: float           # normalised notional USD
    source: Literal["mt5", "dukascopy"]

    @field_validator("bid", "ask")
    @classmethod
    def price_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Price must be > 0, got {v}")
        return round(v, 5)

    @model_validator(mode="after")
    def ask_above_bid(self) -> "UnifiedTick":
        if self.ask <= self.bid:
            raise ValueError(
                f"ask ({self.ask}) must be greater than bid ({self.bid})"
            )
        return self

    @property
    def mid(self) -> float:
        return round((self.bid + self.ask) / 2.0, 5)

    @property
    def spread(self) -> float:
        return round(self.ask - self.bid, 5)


# ── Gold: Liquidity Level ─────────────────────────────────────────────────────

LiqType  = Literal["swing_high", "swing_low", "round_number"]
LiqSide  = Literal["buystops_above", "sellstops_below"]
TimeFrame = Literal["5min", "1H", "4H", "1D"]


class LiquidityLevel(BaseModel):
    """
    A confirmed institutional liquidity level.

    Lifecycle:  formed → [tested N times] → swept → invalidated
    """

    level: float
    liq_type: LiqType
    side: LiqSide               # which side of the level holds the stop orders
    timeframe: TimeFrame        # originating timeframe
    confirmed: bool = False     # True once structure break follows the swing
    swept: bool = False         # True once price has run through the level
    touch_count: int = 0        # number of times price re-tested without sweeping
    swing_size: float = 0.0     # high - low of the originating swing (ATR-relative check)
    formed_at: datetime
    swept_at: Optional[datetime] = None
    bar_time: Optional[datetime] = None   # candle timestamp for merge_asof alignment

    @model_validator(mode="after")
    def swept_requires_time(self) -> "LiquidityLevel":
        if self.swept and self.swept_at is None:
            raise ValueError("swept_at must be set when swept=True")
        return self

    @model_validator(mode="after")
    def infer_side(self) -> "LiquidityLevel":
        """
        Stops always cluster on the obvious side:
          swing_high  → buy stops sit ABOVE the high (breakout traders)
          swing_low   → sell stops sit BELOW the low (breakdown traders)
          round_number → both sides; default to buystops_above
        """
        if self.liq_type == "swing_high":
            object.__setattr__(self, "side", "buystops_above")
        elif self.liq_type == "swing_low":
            object.__setattr__(self, "side", "sellstops_below")
        return self