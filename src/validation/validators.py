from pydantic import BaseModel, field_validator
from typing import Optional
from datetime import datetime

class RawTick(BaseModel):
    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    time_msc: int

    @field_validator("bid", "ask")
    def price_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError(f"Price must be greater than 0, got {v}")
        return v

    @field_validator("symbol")
    def symbol_must_not_be_empty(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Symbol cannot be empty")
        return v

    @field_validator("time_msc")
    def timestamp_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Timestamp must be positive")
        return v


# ── Dukascopy tick (Bronze → Silver) ─────────────────────────────────────────

class DukascopyTick(BaseModel):
    """Validated representation of a single Dukascopy .bi5 tick row."""
    timestamp_utc: datetime
    ask: float
    bid: float
    ask_volume: float
    bid_volume: float
    symbol: str = "XAUUSD"

    @field_validator("ask", "bid")
    def price_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError(f"Price must be greater than 0, got {v}")
        return round(v, 5)

    @field_validator("ask_volume", "bid_volume")
    def volume_must_be_non_negative(cls, v):
        if v < 0:
            raise ValueError(f"Volume cannot be negative, got {v}")
        return v


# ── Unified Silver contract ───────────────────────────────────────────────────

class UnifiedTick(BaseModel):
    """
    Common schema that BOTH MT5 and Dukascopy ticks normalise into.
    This is the input contract for the Gold layer.
    """
    timestamp_utc: datetime   # UTC-aware datetime
    symbol: str
    bid: float
    ask: float
    volume: float             # combined / representative volume
    source: str               # "mt5" | "dukascopy"

    @field_validator("bid", "ask")
    def price_positive(cls, v):
        if v <= 0:
            raise ValueError(f"Price must be > 0, got {v}")
        return round(v, 5)

    @field_validator("source")
    def valid_source(cls, v):
        if v not in ("mt5", "dukascopy"):
            raise ValueError(f"source must be 'mt5' or 'dukascopy', got '{v}'")
        return v