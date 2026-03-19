def make_decision(rsi: float, ema_20: float, bid: float) -> str:
    if rsi < 30 and bid > ema_20:
        return "BUY"
    elif rsi > 70 and bid < ema_20:
        return "SELL"
    else:
        return "HOLD"