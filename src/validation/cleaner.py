import asyncpg
import asyncio
import pandas as pd
import ta
from dotenv import load_dotenv
from src.validation.validators import RawTick
from pydantic import ValidationError
import os

load_dotenv()

async def get_db_connection():
    return await asyncpg.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT")),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )

async def fetch_unprocessed_ticks(conn):
    rows = await conn.fetch("""
        SELECT l.* FROM landing_ticks l
        LEFT JOIN cleaned_ticks c ON l.time_msc = c.time_msc
        WHERE c.time_msc IS NULL
        ORDER BY l.time_msc ASC
        LIMIT 100
    """)
    return rows

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["ema_20"] = ta.trend.ema_indicator(df["bid"], window=20)
    df["rsi"] = ta.momentum.rsi(df["bid"], window=14)
    df["spread"] = df["ask"] - df["bid"]
    return df

async def clean_and_store(conn, tick_row, ema_20, rsi, spread):
    await conn.execute("""
        INSERT INTO cleaned_ticks
            (symbol, bid, ask, spread, rsi, ema_20, tick_time)
        VALUES
            ($1, $2, $3, $4, $5, $6, to_timestamp($7::double precision / 1000))
        ON CONFLICT DO NOTHING
    """,
        tick_row["symbol"],
        tick_row["bid"],
        tick_row["ask"],
        spread,
        rsi,
        ema_20,
        tick_row["time_msc"]
    )

async def run_cleaner():
    conn = await get_db_connection()
    print("Silver layer cleaner running...")

    try:
        rows = await fetch_unprocessed_ticks(conn)

        if not rows:
            print("No new ticks to process")
            return

        # validate each row with pydantic
        valid_ticks = []
        for row in rows:
            try:
                tick = RawTick(
                    symbol=row["symbol"],
                    bid=row["bid"],
                    ask=row["ask"],
                    last=row["last"],
                    volume=row["volume"],
                    time_msc=row["time_msc"]
                )
                valid_ticks.append(dict(tick))
            except ValidationError as e:
                print(f"Invalid tick skipped: {e}")
                continue

        if not valid_ticks:
            print("No valid ticks after validation")
            return

        # calculate indicators
        df = pd.DataFrame(valid_ticks)
        df = calculate_indicators(df)

        # write to cleaned_ticks
        saved = 0
        for _, row in df.iterrows():
            if pd.isna(row["rsi"]) or pd.isna(row["ema_20"]):
                continue  # skip rows without enough data for indicators
            await clean_and_store(
                conn,
                row,
                ema_20=row["ema_20"],
                rsi=row["rsi"],
                spread=row["spread"]
            )
            saved += 1

        print(f"Silver layer: {saved} ticks cleaned and stored")

    finally:
        await conn.close()

if __name__ == "__main__":
    # mock data for testing without MT5
    mock_ticks = [
        {"symbol": "XAUUSD", "bid": 5020.95, "ask": 5021.11, 
         "last": 5020.95, "volume": 1.0, "time_msc": 1773446279879 + i * 1000}
        for i in range(50)
    ]
    print("Testing with mock data...")
    for tick in mock_ticks:
        try:
            validated = RawTick(**tick)
            print(f"Valid tick: BID {validated.bid} ASK {validated.ask}")
        except ValidationError as e:
            print(f"Invalid: {e}")
    
    asyncio.run(run_cleaner())