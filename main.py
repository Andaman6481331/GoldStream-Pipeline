import asyncio
import MetaTrader5 as mt5
from src.ingestion.mt5_client import connect_mt5, fetch_tick
from src.ingestion.raw_writer import write_raw_tick
from src.validation.cleaner import run_cleaner
from src.bot.audit_logger import run_gold_layer
from dotenv import load_dotenv

load_dotenv()

async def run():
    if not connect_mt5():
        return

    print("GoldStream Pipeline running...")
    tick_count = 0

    while True:
        tick = fetch_tick()
        if tick:
            await write_raw_tick(tick)
            tick_count += 1
            print(f"Tick {tick_count} saved to Bronze")

            # run silver cleaner every 25 ticks
            if tick_count % 25 == 0:
                print("Running Silver layer cleaner...")
                await run_cleaner()
                print("--- Running Gold layer ---")
                await run_gold_layer()

        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(run())