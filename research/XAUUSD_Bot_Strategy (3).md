# XAUUSD Liquidity Sweep Bot — Full Strategy Document
> All logic is fully defined. No remaining [RESEARCH] items.
> `[BACKTEST]` = value to be confirmed via backtesting on historical XAUUSD data.

---

## Overview

A two-trade pipeline that exploits liquidity sweeps and market structure breaks on gold (XAUUSD).

- **Trade 1 (scout):** small lot, market order, fires immediately on BOS/CHoCH confirmation. Trail-only exit, no TP.
- **Trade 2 (sniper):** larger lot, limit order at FVG midpoint, only if Trade 1 SL is hit and all qualification gates pass. Optional structural TP at nearest Tier 1/2 liquidity level.
- **No fixed TP on Trade 1** — trailing SL only, let sweep reversals run fully.
- **Optional structural TP on Trade 2** — only placed if R:R ≥ MIN_TP_RR, otherwise trail-only.

---

## Tick Data Structure

```python
tick = {
  # raw price feed
  "bid":        float,
  "ask":        float,
  "last":       float,
  "spread":     float,
  "volume":     int,
  "timestamp":  int,       # unix milliseconds

  # derived
  "mid":        float,     # (bid + ask) / 2
  "session":    str,       # "london" | "newyork" | "asian" | "off"

  # candle state — built from ticks by bot, NOT from broker rows
  "bar_1m":  { "open": float, "high": float, "low": float, "close": float, "closed": bool },
  "bar_15m": { "open": float, "high": float, "low": float, "close": float, "closed": bool },
  "bar_4h":  { "open": float, "high": float, "low": float, "close": float, "closed": bool },

  # session levels — updated at each session boundary
  "session_levels": {
    "prev_day_high":        float,
    "prev_day_low":         float,
    "current_session_high": float,
    "current_session_low":  float,
    "prev_session_high":    float,
    "prev_session_low":     float,
  },

  # ATR — recomputed on each relevant closed candle
  "atr15_15m": float,   # ATR(15) on 15m — R_dynamic, sweep tolerance, Point 1/2
  "atr20_1m":  float,   # ATR(20) on 1m  — FVG size filter, displacement check

  # confirmed swing point history — appended by fractal detector on each closed 15m bar
  "confirmed_swing_highs_15m": list[SwingPoint],  # { price, candle_index }
  "confirmed_swing_lows_15m":  list[SwingPoint],

  # trade state
  "trade1": {
    "active":            bool,
    "entry":             float,
    "entry_candle_high": float,   # 1m candle high at entry moment — for Point 1
    "entry_candle_low":  float,   # 1m candle low at entry moment  — for Point 1 short
    "sl":                float,
    "risk_distance":     float,   # abs(entry - sl), calculated once at entry
    "lot":               float,
    "point1":            float,
    "point2":            float,
    "point1_hit":        bool,
    "point2_hit":        bool,
    "trailing":          bool,
    "tp":                None,    # always None for Trade 1
    "bos_time":          int,     # unix ms
    "bos_direction":     str,     # "bull" | "bear"
    "bos_type":          str,     # "bos" | "choch"
    "sweep_candle_low":  float,
    "sweep_candle_high": float,
    "sweep_tier":        int,     # 1 | 2 | 3
    "r_dynamic":         int,     # R at BOS time — used for Gate 3 timing
  },
  "trade2": {
    "pending":    bool,
    "entry":      float,   # FVG midpoint — limit order price
    "sl":         float,
    "tp":         float,   # structural TP if R:R qualifies, else None
    "lot":        float,
    "fvg_low":    float,
    "fvg_high":   float,
    "multiplier": float,
    "expire_at":  int,     # unix ms — cancel if not filled by this time
  }
}
```

---

## Bot State Machine (runs on every tick)

```
on every tick:
  1. update open bar (1m, 15m, 4H) with new price

  2. if session boundary crossed:
       update session_levels

  3. if 1m bar just closed:
       recompute atr20_1m
       run FVG scanner on last 3 closed 1m candles
       run 1m swing detection for trailing SL (L=R=3 fixed)

  4. if 15m bar just closed:
       recompute atr15_15m
       compute R_dynamic from atr15_15m
       run swing detection on 15m (L=5, R=R_dynamic)
       append new confirmed swings to confirmed_swing_highs/lows_15m
       update structural node state (HH, LL, StrongLow, StrongHigh)
       rebuild liquidity levels (refreshes EQH/EQL clusters)
       check for BOS/CHoCH on 15m

  5. if 4H bar just closed:
       recalculate market_bias (L=R=5 fixed)

  6. if trade1 active:
       check if price hit point1  → move SL to breakeven
       check if price hit point2  → activate trailing SL
       if trailing → update SL to last confirmed 1m swing point (one-way only)
       if price hit SL → close trade1, trigger Phase 3 evaluation

  7. if trade2 pending:
       if price reached FVG midpoint → broker fills entire lot at once
       if FVG refilled → cancel order
       if timeout expired → cancel order
       if market_bias flipped → cancel order

  8. if trade2 active:
       same Point 1 / Point 2 / trailing SL logic as trade1
       if tp set and price reached tp → close trade2
```

---

## Background — 4H Market Bias (always running)

```python
# L=R=5 fixed — structural context, not entry timing
# bull: HH+HL sequence | bear: LH+LL | neutral: insufficient data
market_bias = calc_structure(candles_4h, L=5, R=5)
# no trading while market_bias == "neutral"
```

---

## Phase 1 — BOS / CHoCH Detection

### Step 1 — Dynamic R Calculation

```python
atr15_15m = calculate_atr(candles_15m, period=15)
R_dynamic = clamp(round(k / atr15_15m), 2, 5)

# k: calibration constant in same price units as ATR
# k [BACKTEST] — suggested: 3 × historical_average_atr15_15m
# example: average atr15_15m = 8.0 → k = 24.0
#   ATR=12 → R=2  (high vol — confirm faster)
#   ATR=8  → R=3  (normal vol)
#   ATR=6  → R=4  (low vol)
#   ATR=4  → clamped to R=5 (very low vol)
```

L=5 always fixed. Only R adapts to volatility.
Gate 3 timing uses `r_dynamic` stored at BOS time — not current R.

### Step 2 — Swing Point Detection (Williams Fractal)

```python
# Swing High at index i (L=5, R=R_dynamic):
is_swing_high = (
    candles[i].high > max(c.high for c in candles[i-5 : i])
    and
    candles[i].high > max(c.high for c in candles[i+1 : i+R_dynamic+1])
)

# Swing Low at index i:
is_swing_low = (
    candles[i].low < min(c.low for c in candles[i-5 : i])
    and
    candles[i].low < min(c.low for c in candles[i+1 : i+R_dynamic+1])
)
```

Confirmation lag = R_dynamic × 15 min. Never use unconfirmed candles.
Initialization: needs L + R_max + 1 = 11 closed 15m candles minimum before first swing.

### Step 3 — Structural Node State Machine

```python
state = {
    "trend":       None,   # "bull" | "bear" | None (unknown at start)
    "HH":          None,   # highest confirmed swing high
    "LL":          None,   # lowest confirmed swing low
    "strong_low":  None,   # SL that preceded current HH
    "strong_high": None,   # SH that preceded current LL
}

# BULLISH: new confirmed SH > HH
#   → HH = new SH
#   → strong_low = most recent confirmed SL before this SH

# BEARISH: new confirmed SL < LL
#   → LL = new SL
#   → strong_high = most recent confirmed SH before this SL
```

### Step 4 — BOS and CHoCH Calculation

Close-only confirmation. Wick cross alone = not a BOS/CHoCH, but triggers FVG scan.

```
BULLISH:
  BOS:   Close_i > HH
           → stay bull · strong_low = SL before break · fire T1 long  · bos_type="bos"
  CHoCH: Close_i < strong_low
           → flip bear · strong_high = highest before break · fire T1 short · bos_type="choch"

BEARISH:
  BOS:   Close_i < LL
           → stay bear · strong_high = SH before break · fire T1 short · bos_type="bos"
  CHoCH: Close_i > strong_high
           → flip bull · strong_low = lowest before break · fire T1 long · bos_type="choch"
```

Direction must align with 4H market_bias. Counter-trend signals ignored.

---

## Phase 2 — Trade 1 (scout entry)

### Entry

```python
entry             = market_order_now()
entry_candle_high = current_1m_candle.high   # snapshot at entry moment
entry_candle_low  = current_1m_candle.low
lot               = base_lot                  # e.g. 0.01
sl                = calculate_trade1_sl()
tp                = None                      # Trade 1 never has a TP
```

### Trade 1 SL Placement

SL = lowest/highest wick extreme across all 15m candles in the sweep zone
(between StrongLow/StrongHigh and the BOS candle), plus a buffer.

```python
# Long:
sweep_zone   = candles_15m[strong_low_index : bos_candle_index + 1]
sweep_candle = min(sweep_zone, key=lambda c: c.low)
sl           = sweep_candle.low - T1_BUFFER_PIPS

# Short:
sweep_zone   = candles_15m[strong_high_index : bos_candle_index + 1]
sweep_candle = max(sweep_zone, key=lambda c: c.high)
sl           = sweep_candle.high + T1_BUFFER_PIPS

# T1_BUFFER_PIPS [BACKTEST] — suggested: 3–5 pips
```

`min()`/`max()` across all sweep zone candles handles multi-candle sweeps automatically.

### Risk Distance

```python
risk_distance = abs(entry - sl)   # stored once at entry
```

### Point 1 — Move SL to Breakeven

Structural trigger (first 1m close beyond entry candle extreme),
floored by ATR-relative minimum distance to prevent spread noise triggers.

```python
# Long:
point1 = max(entry_candle_high, entry + atr15_15m * P1_ATR_FACTOR)
# Short:
point1 = min(entry_candle_low,  entry - atr15_15m * P1_ATR_FACTOR)

# Trigger:
if long  and price >= point1: sl = entry · point1_hit = True
if short and price <= point1: sl = entry · point1_hit = True

# P1_ATR_FACTOR [BACKTEST] — suggested: 0.3
```

### Point 2 — Activate Trailing SL

ATR-based, guaranteed minimum gap from Point 1.

```python
# Long:
point2 = max(entry + atr15_15m * P2_ATR_FACTOR,
             point1 + atr15_15m * P2_P1_MIN_GAP)
# Short:
point2 = min(entry - atr15_15m * P2_ATR_FACTOR,
             point1 - atr15_15m * P2_P1_MIN_GAP)

# Trigger:
if long  and price >= point2: activate_trailing_sl() · point2_hit = True
if short and price <= point2: activate_trailing_sl() · point2_hit = True

# P2_ATR_FACTOR [BACKTEST] — suggested: 1.0–1.5
# P2_P1_MIN_GAP [BACKTEST] — suggested: 0.3
```

### Trailing SL Mechanic

Fractal-based on confirmed 1m swing points. L=R=3 fixed on 1m.

```python
# Long: on each new confirmed 1m swing low
if new_1m_swing_low_confirmed:
    candidate = new_1m_swing_low.low - TRAIL_BUFFER_PIPS
    if candidate > current_sl:    # one-way only — never widen
        sl = candidate

# Short: mirror using confirmed 1m swing highs

# TRAIL_BUFFER_PIPS [BACKTEST] — suggested: 2–3 pips
```

### If Trade 1 Hits SL → proceed to Phase 3

---

## Phase 3 — Trade 2 Qualification

Only reached if Trade 1 SL was hit.
All hard gates must pass. High confidence score does NOT override a failed gate.

---

### Layer 1 — Hard Gates

#### Gate 1 — 1m FVG exists, clean, and shows real displacement

```python
# Find 3-candle gap on closed 1m candles:
for i in range(1, len(candles_1m) - 1):
    fvg_low  = candles_1m[i-1].high
    fvg_high = candles_1m[i+1].low
    if fvg_high > fvg_low:
        fvg = FVG(low=fvg_low, high=fvg_high, impulse_candle=candles_1m[i])

if not fvg:          abort()   # no gap found
if fvg.is_refilled:  abort()   # gap already traded through

# Displacement check — impulse candle body must be a real move
impulse_body = abs(fvg.impulse_candle.close - fvg.impulse_candle.open)
if impulse_body < atr20_1m * DISPLACEMENT_FACTOR:
    abort()   # body too small — spread artifact or noise, not a real move

# DISPLACEMENT_FACTOR [BACKTEST] — suggested: 0.5
```

FVG must originate from the impulse candle that caused (or directly followed) the BOS.
Also applies to wick-only cross scenarios — scan FVG on the wick candle's impulse move.

#### Gate 2 — FVG size above noise threshold

```python
min_fvg_size = max(atr20_1m * FVG_ATR_MULTIPLIER, ABSOLUTE_MIN_PIPS)
if (fvg.high - fvg.low) < min_fvg_size:
    abort()

# FVG_ATR_MULTIPLIER [BACKTEST] — suggested: 0.10–0.20
# ABSOLUTE_MIN_PIPS  [BACKTEST] — suggested: 3 pips
```

#### Gate 3 — 15m structure still intact (time-conditional)

```python
# Use R stored at BOS time — not current R_dynamic
confirmation_lag_ms = trade1.r_dynamic * 15 * 60 * 1000

if (now - trade1.bos_time) >= confirmation_lag_ms:
    if not is_15m_structure_intact(direction=bos_direction):
        abort()
# if elapsed < lag: skip — swing history unchanged, scoring provides protection
```

#### Gate 4 — Liquidity sweep confirmed

**Sweep tolerance (ATR-based):**

```python
sweep_tolerance = max(atr15_15m * SWEEP_ATR_FACTOR, SWEEP_ABSOLUTE_MIN_PIPS)
# SWEEP_ATR_FACTOR        [BACKTEST] — suggested: 0.10–0.20
# SWEEP_ABSOLUTE_MIN_PIPS [BACKTEST] — suggested: 2 pips
```

**Liquidity level tiers:**

| Tier | Level | Score contribution |
|---|---|---|
| 1 | Previous day high / low | +2 to sweep score |
| 2 | Current + previous session high / low | +1 to sweep score |
| 3 | StrongHigh / StrongLow from state machine | +0 (passes gate only) |
| 3 | Equal highs / equal lows clusters | +0 (passes gate only) |

**Session boundaries:** Asian 00–09 UTC · London 08–17 UTC · NY 13–22 UTC

**Equal Highs / Equal Lows Detection:**

Cluster of ≥ MIN_CLUSTER_SIZE confirmed 15m swing points within ATR-relative tolerance.
Cluster average price = level price. Deduplication keeps strongest cluster per zone.
Consumed levels (price has closed beyond and stayed beyond) are filtered out.

```python
@dataclass
class Level:
    price:    float
    tier:     int        # 1 | 2 | 3
    strength: int = 1    # swing point count (EQH/EQL only)

def find_equal_highs(confirmed_swing_highs, tolerance) -> list[Level]:
    equal_levels = []
    for i, sh_a in enumerate(confirmed_swing_highs):
        cluster = [sh_a] + [
            sh_b for sh_b in confirmed_swing_highs[i+1:]
            if abs(sh_a.price - sh_b.price) <= tolerance
        ]
        if len(cluster) >= MIN_CLUSTER_SIZE:
            equal_levels.append(Level(
                price    = sum(s.price for s in cluster) / len(cluster),
                tier     = 3,
                strength = len(cluster)
            ))
    return deduplicate_clusters(equal_levels, tolerance)

def deduplicate_clusters(levels, tolerance) -> list[Level]:
    seen = []
    for level in sorted(levels, key=lambda l: -l.strength):
        if not any(abs(level.price - s.price) <= tolerance for s in seen):
            seen.append(level)
    return seen

def is_level_consumed(level, candles_15m, sweep_tolerance) -> bool:
    recent = candles_15m[-CONSUMPTION_LOOKBACK:]
    if level.direction == "high":
        return all(c.close > level.price + sweep_tolerance for c in recent)
    return all(c.close < level.price - sweep_tolerance for c in recent)

# MIN_CLUSTER_SIZE = 2
# EQUAL_HL_ATR_FACTOR        [BACKTEST] — suggested: 0.20–0.30
# EQUAL_HL_LOOKBACK_CANDLES  [BACKTEST] — suggested: 50–100 closed 15m candles
# CONSUMPTION_LOOKBACK       [BACKTEST] — suggested: 5 candles
```

**Build all levels:**

```python
def build_liquidity_levels(session_levels, state_15m,
                            confirmed_swing_highs, confirmed_swing_lows,
                            atr15_15m, candles_15m) -> list[Level]:
    tolerance = atr15_15m * EQUAL_HL_ATR_FACTOR
    levels = []

    # Tier 1
    levels += [Level(session_levels.prev_day_high, tier=1),
               Level(session_levels.prev_day_low,  tier=1)]
    # Tier 2
    levels += [Level(session_levels.current_session_high, tier=2),
               Level(session_levels.current_session_low,  tier=2),
               Level(session_levels.prev_session_high,    tier=2),
               Level(session_levels.prev_session_low,     tier=2)]
    # Tier 3 — structural nodes
    levels += [Level(state_15m.strong_low,  tier=3),
               Level(state_15m.strong_high, tier=3)]
    # Tier 3 — equal HL clusters
    recent_highs = confirmed_swing_highs[-EQUAL_HL_LOOKBACK_CANDLES:]
    recent_lows  = confirmed_swing_lows[-EQUAL_HL_LOOKBACK_CANDLES:]
    eqh = find_equal_highs(recent_highs, tolerance)
    eql = find_equal_lows(recent_lows,   tolerance)
    levels += [l for l in eqh + eql
               if not is_level_consumed(l, candles_15m, sweep_tolerance)]
    return levels
```

**Sweep check:**

```python
def is_liquidity_sweep(sweep_zone_candles, direction,
                        levels, sweep_tolerance) -> tuple[bool, int]:
    for level in sorted(levels, key=lambda l: l.tier):   # tier 1 first
        for candle in sweep_zone_candles:
            if direction == "bull":
                if (candle.low  <= level.price + sweep_tolerance
                        and candle.close > level.price):
                    return True, level.tier
            if direction == "bear":
                if (candle.high >= level.price - sweep_tolerance
                        and candle.close < level.price):
                    return True, level.tier
    return False, None

# In Phase 3:
swept, sweep_tier = is_liquidity_sweep(
    sweep_zone_candles, bos_direction,
    build_liquidity_levels(...), sweep_tolerance
)
if not swept:
    abort()
trade1.sweep_tier = sweep_tier
```

---

### Layer 2 — Confidence Scoring

#### Sweep Strength Score (max 5)

| Condition | Points |
|---|---|
| Swept Tier 1 level (daily high/low) | +2 |
| Swept Tier 2 level (session high/low) | +1 |
| Sweep wick > 50% of candle body | +1 |
| Sweep occurred during London or NY session | +1 |
| Signal is CHoCH (trend flip, not BOS continuation) | +1 |

Tier 1 and Tier 2 are mutually exclusive. Tier 3 passes gate, contributes 0 points.
EQH/EQL cluster strength ≥ 3 is logged but does not add extra score.
Score capped at 5.

#### FVG Quality Score (max 5)

| Condition | Points |
|---|---|
| FVG width > 15 pips | +2 |
| FVG formed on the impulse candle itself | +1 |
| FVG sits inside a 4H order block zone | +1 |
| FVG unfilled for > 3 closed 1m candles before entry | +1 |

#### Multiplier Calculation

```python
total_score = min(sweep_score, 5) + min(fvg_score, 5)   # 0–10
if total_score < MIN_SCORE:
    abort()
re_entry_multiplier = (total_score / 10) * 5             # 0.0–5.0
lot = base_lot * max(1, floor(re_entry_multiplier))

# MIN_SCORE [BACKTEST] — suggested: 4
```

| Score | Multiplier | Lot (base 0.01) |
|---|---|---|
| 4 | 2.0 | 0.02 |
| 6 | 3.0 | 0.03 |
| 8 | 4.0 | 0.04 |
| 10 | 5.0 | 0.05 |

---

## Phase 4 — Trade 2 Entry (sniper)

### Entry

```python
entry = (fvg.low + fvg.high) / 2              # limit order — broker fills full lot at once
sl    = fvg.low  - T2_BUFFER_PIPS             # long
      = fvg.high + T2_BUFFER_PIPS             # short
lot   = base_lot * max(1, floor(re_entry_multiplier))

# T2_BUFFER_PIPS [BACKTEST] — suggested: 2–3 pips
```

### Trade 2 TP — Optional Structural TP

Trade 1 has no TP — trail only, let sweep reversals run fully.
Trade 2 gets a structural TP at the nearest Tier 1/2 level in trade direction,
but only if it represents minimum qualifying R:R.

```python
tp_candidates = [
    l for l in liquidity_levels
    if l.tier in (1, 2)
    and (l.price > entry if direction == "bull" else l.price < entry)
]

if tp_candidates:
    nearest = min(tp_candidates, key=lambda l: abs(l.price - entry))
    if abs(nearest.price - entry) >= abs(entry - sl) * MIN_TP_RR:
        tp = nearest.price   # structural TP — minimum R:R satisfied
    else:
        tp = None            # level too close — trail only
else:
    tp = None                # no qualifying level — trail only

# MIN_TP_RR [BACKTEST] — suggested: 2.0 (minimum 1:2 R:R)
```

### Order Validity

```python
if price_refills_fvg():        cancel_order()
if time_elapsed > T2_TIMEOUT:  cancel_order()
if market_bias_flips():        cancel_order()

# T2_TIMEOUT [BACKTEST] — suggested: 10 minutes
```

### After Fill

Same Point 1 / Point 2 / trailing SL logic as Trade 1.
Risk distance recalculated from T2 entry to T2 SL.
If TP is set, close at TP before trail would trigger.

---

## All Price Scenarios

| # | What price does | T1 result | T2 result |
|---|---|---|---|
| 1 | BOS → straight run, no SL hit | trails to big win | never triggered |
| 2 | BOS → Point 1 (BE) → reverses → stopped | breakeven | never triggered |
| 3 | BOS → Point 1 → Point 2 → reverses | small locked win | never triggered |
| 4 | BOS → Point 1 → Point 2 → full trail run | maximum win | never triggered |
| 5 | BOS → T1 SL → FVG → trend resumes → T2 TP hit | small loss | clean win at structural TP |
| 6 | BOS → T1 SL → FVG → trend resumes → T2 trails | small loss | big trail win |
| 7 | BOS → T1 SL → FVG → T2 SL also hit | small loss | loss (worst case) |
| 8 | BOS → T1 SL → no FVG forms | small loss | blocked |
| 9 | BOS → T1 SL → FVG → hard gate fails | small loss | blocked correctly |
| 10 | BOS → T1 SL → FVG → score < MIN_SCORE | small loss | blocked (low confidence) |
| 11 | T2 limit placed → FVG refills before fill | small loss | cancelled |

**Position sizing rule:** T1 + T2 combined max loss ≤ 1% of account.

---

## Drawbacks and Risks

- **False BOS:** body-close confirmation + 15m timeframe reduces noise. 1m is never a BOS trigger.
- **Dynamic R recalculates each bar:** Gate 3 uses `r_dynamic` stored at BOS time — not current R.
- **Double loss (scenario 7):** contained only by position sizing, not logic.
- **FVG refills fast on gold:** displacement check + ATR size filter reduce fake entries. T2_TIMEOUT must be tight.
- **Tier 3 sweep only:** passes gate, 0 score → low/no multiplier → correct small lot sizing.
- **EQH/EQL clusters:** consumed levels filtered before each sweep check. Prevents stale level trades.
- **Trade 2 TP may not exist:** if no Tier 1/2 level qualifies, T2 trails — this is correct behaviour, not a failure.

---

## Complete Parameters Reference

### Fixed (not tunable)

| Parameter | Value | Reason |
|---|---|---|
| 4H fractal L | 5 | structural context — stable |
| 4H fractal R | 5 | structural context — stable |
| 15m fractal L | 5 | lookback fixed — only R adapts |
| R_dynamic clamp | 2–5 | bounds on confirmation window |
| 1m fractal L | 3 | faster trail response |
| 1m fractal R | 3 | faster trail response |
| MIN_CLUSTER_SIZE | 2 | minimum for a real equal level |
| Trade 1 TP | None | trail-only always |
| Session: Asian | 00–09 UTC | — |
| Session: London | 08–17 UTC | — |
| Session: NY | 13–22 UTC | — |

### Needs backtesting

| Parameter | Suggested starting value |
|---|---|
| `k` (R_dynamic constant) | 3 × avg atr15_15m |
| `T1_BUFFER_PIPS` | 3–5 pips |
| `P1_ATR_FACTOR` | 0.3 |
| `P2_ATR_FACTOR` | 1.0–1.5 |
| `P2_P1_MIN_GAP` | 0.3 |
| `TRAIL_BUFFER_PIPS` | 2–3 pips |
| `DISPLACEMENT_FACTOR` | 0.5 |
| `FVG_ATR_MULTIPLIER` | 0.10–0.20 |
| `ABSOLUTE_MIN_PIPS` | 3 pips |
| `MIN_SCORE` | 4 |
| `MIN_TP_RR` | 2.0 |
| `T2_BUFFER_PIPS` | 2–3 pips |
| `T2_TIMEOUT` | 10 minutes |
| `SWEEP_ATR_FACTOR` | 0.10–0.20 |
| `SWEEP_ABSOLUTE_MIN_PIPS` | 2 pips |
| `EQUAL_HL_ATR_FACTOR` | 0.20–0.30 |
| `EQUAL_HL_LOOKBACK_CANDLES` | 50–100 |
| `CONSUMPTION_LOOKBACK` | 5 candles |
| Sessions to include in sweep | current + 1 previous |

---

## Future Work

These items are intentionally excluded from the current build.
Add only after the core system is backtested and validated.

| Item | Reason deferred |
|---|---|
| **News/event filter** | Detecting high-impact events (NFP, CPI, FOMC) requires an economic calendar feed. Add after backtesting identifies which losing trades cluster around news events. Block trading N minutes before/after scheduled events. |
| **HTF Order Block (OB) detection** | FVG quality score awards +1 for sitting inside a 4H OB zone. Currently this requires manual identification. Automate after core system is stable. |
| **Multi-session equal HL refresh** | Currently EQH/EQL uses a fixed lookback. A smarter approach refreshes levels per session open. Complexity not justified until backtesting shows it matters. |
| **Adaptive MIN_SCORE by session** | Lower MIN_SCORE threshold during London/NY (higher quality setups), higher during Asian (more noise). Defer until base MIN_SCORE is calibrated. |

---

## Development Phases

```
Phase 1 — build now
  Candle builder from tick data (1m, 15m, 4H)
  ATR calculator (atr15_15m, atr20_1m)
  Session level tracker (daily + session high/low + boundary detection)
  FVG scanner (3-candle gap + ATR size filter + displacement check)
  re_entry_multiplier scoring engine
  Trailing SL logic (fractal 1m L=R=3, one-way)

Phase 2
  R_dynamic calculator
  Williams Fractal swing detector (15m L=5 R=R_dynamic · 4H L=R=5)
  Structural node state machine (HH / LL / StrongLow / StrongHigh)
  BOS/CHoCH detection (close-only confirmation)
  4H market_bias calculator
  Trade 1 SL calculator (sweep extreme finder)
  Equal highs/lows detector (ATR-tolerance clustering + deduplication)
  is_liquidity_sweep (tiered levels + ATR tolerance)
  Trade 2 structural TP finder

Phase 3
  Connect Phase 1 + Phase 2 into full pipeline
  End-to-end integration testing on historical data

Phase 4
  Backtesting on historical XAUUSD tick/candle data
  Resolve all [BACKTEST] parameter values
  Risk and drawdown analysis
  Validate: T1 + T2 combined max loss ≤ 1% of account
  Add future work items as data justifies
```
