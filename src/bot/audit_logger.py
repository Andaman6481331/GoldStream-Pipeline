import asyncpg
import asyncio
from dotenv import load_dotenv
from src.bot.strategy import make_decision
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

async def run_gold_layer():
    conn = await get_db_connection()
    print("Gold layer running...")

    try:
        rows = await conn.fetch("""
            SELECT c.* FROM cleaned_ticks c
            LEFT JOIN trade_decisions t ON c.tick_time = t.tick_time
            WHERE t.tick_time IS NULL
            AND c.rsi IS NOT NULL
            AND c.ema_20 IS NOT NULL
            ORDER BY c.tick_time ASC
        """)

        if not rows:
            print("No new cleaned ticks to process")
            return

        saved = 0
        for row in rows:
            decision = make_decision(
                rsi=row["rsi"],
                ema_20=row["ema_20"],
                bid=row["bid"]
            )

            await conn.execute("""
                INSERT INTO trade_decisions
                    (symbol, decision, bid, ask, rsi, ema_20, spread, tick_time)
                VALUES
                    ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT DO NOTHING
            """,
                row["symbol"],
                decision,
                row["bid"],
                row["ask"],
                row["rsi"],
                row["ema_20"],
                row["spread"],
                row["tick_time"]
            )
            saved += 1
            print(f"Decision: {decision} | BID: {row['bid']} RSI: {row['rsi']:.2f}")

        print(f"Gold layer: {saved} decisions logged")

    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(run_gold_layer())