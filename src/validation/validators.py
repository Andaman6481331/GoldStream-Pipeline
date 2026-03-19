from pydantic import BaseModel, field_validator
from typing import Optional

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