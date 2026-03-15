import asyncpg
import asyncio
from dotenv import load_dotenv
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

async def write_raw_tick(tick: dict):
    conn = await get_db_connection()
    try:
        await conn.execute("""
            INSERT INTO landing_ticks 
                (symbol, bid, ask, last, volume, time_msc)
            VALUES 
                ($1, $2, $3, $4, $5, $6)
        """,
            tick["symbol"],
            tick["bid"],
            tick["ask"],
            tick["last"],
            tick["volume"],
            tick["time_msc"]
        )
        print(f"Saved to landing: BID {tick['bid']} ASK {tick['ask']}")
    finally:
        await conn.close()