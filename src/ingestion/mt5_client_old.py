import MetaTrader5 as mt5
from datetime import datetime
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()

def connect_mt5():
    if not mt5.initialize():
        print(f"MT5 initialize failed: {mt5.last_error()}")
        return False
    
    login = int(os.getenv("MT5_LOGIN"))
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")

    if not mt5.login(login, password=password, server=server):
        print(f"MT5 login failed: {mt5.last_error()}")
        return False

    print(f"Connected to MT5: {mt5.account_info().server}")
    return True

def fetch_tick():
    tick = mt5.symbol_info_tick("XAUUSD")
    if tick is None:
        print("Failed to get tick")
        return None
    
    return {
        "symbol": "XAUUSD",
        "bid": tick.bid,
        "ask": tick.ask,
        "last": tick.last,
        "volume": tick.volume,
        "time_msc": tick.time_msc
    }

async def run():
    if not connect_mt5():
        return
    
    print("Starting tick feed...")
    while True:
        tick = fetch_tick()
        if tick:
            print(f"BID: {tick['bid']} ASK: {tick['ask']} TIME: {tick['time_msc']}")
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(run())