# XAUUSD Liquidity Sweep Bot — Full Strategy Document
> Legend: `[RESEARCH]` = needs conceptual research/decision | `[BACKTEST]` = value to be found via backtesting

---

## Overview

A two-trade pipeline that exploits liquidity sweeps and market structure breaks on gold (XAUUSD).

- **Trade 1 (scout):** small lot, market order, fires immediately on BOS/CHoCH confirmation
- **Trade 2 (sniper):** larger lot, limit order at FVG midpoint, only if Trade 1 SL is hit and all qualification gates pass
- **No fixed TP on either trade** — trailing SL only on both

---

## Tick Data Structure

Every tick received by the bot must carry and compute the following fields:

```python
tick = {
  # raw price feed
  "bid":        float,     # best bid price
  "ask":        float,     # best ask price
  "last":       float,     # last traded price
  "spread":     float,     # ask - bid
  "volume":     int,       # tick volume
  "timestamp":  int,       # unix milliseconds

  # derived — computed by bot on each tick
  "mid":        float,     # (bid + ask) / 2
  "session":    str,       # "london" | "newyork" | "asian" | "off"

  # candle state — bot builds from ticks, NOT from broker data rows
  "bar_1m":  { "open": float, "high": float, "low": float, "close": float, "closed": bool },
  "bar_15m": { "open": float, "high": float, "low": float, "close": float, "closed": bool },
  "bar_4h":  { "open": float, "high": float, "low": float, "close": float, "closed": bool },

  # session levels — updated at session open
  "session_levels": {
    "prev_day_high":     float,
    "prev_day_low":      float,
    "current_session_high": float,
    "current_session_low":  float,
    "prev_session_high": float,
    "prev_session_low":  float,
  },

  # trade state — updated by bot logic
  "trade1": {
    "active":            bool,
    "entry":             float,
    "sl":                float,
    "risk_distance":     float,   # abs(entry - sl), calculated once at entry
    "lot":               float,
    "point1":            float,   # price level for BE trigger
    "point2":            float,   # price level for trail activation
    "point1_hit":        bool,
    "point2_hit":        bool,
    "trailing":          bool,
    "bos_time":          int,     # unix ms when BOS fired
    "bos_direction":     str,     # "bull" | "bear"
    "sweep_candle_low":  float,   # for longs: lowest wick in sweep zone
    "sweep_candle_high": float,   # for shorts: highest wick in sweep zone
    "sweep_tier":        int,     # 1 | 2 | 3 — tier of level swept
  },
  "trade2": {
    "pending":     bool,
    "entry":       float,   # FVG midpoint
    "sl":          float,
    "lot":         float,
    "fvg_low":     float,
    "fvg_high":    float,
    "multiplier":  float,
    "expire_at":   int,     # unix ms timeout
  }
}
```

---

## Bot State Machine (runs on every tick)

```
on every tick:
  1. update open bar (1m, 15m, 4H) with new price

  2. if session boundary crossed:
       update session_levels (prev_day_high/low, session_high/low)

  3. if 1m bar just closed:
       run FVG scanner on last 3 closed 1m candles
       run swing detection on 1m candles (for trailing SL reference)

  4. if 15m bar just closed:
       run swing detection on 15m candles (L=R=5)
       update 15m structural node state (HH, LL, StrongLow, StrongHigh)
       check for BOS/CHoCH on 15m

  5. if 4H bar just closed:
       recalculate market_bias from 4H structure

  6. if trade1 active:
       check if price hit point1  → move SL to breakeven
       check if price hit point2  → activate trailing SL
       check if trailing active   → update SL to last confirmed 1m structure point
       check if price hit SL      → close trade1, trigger Phase 3 evaluation

  7. if trade2 pending:
       check if price reached FVG midpoint → open trade2
       check if FVG is still valid (not refilled)
       check if timeout expired   → cancel trade2
       check if market_bias flipped → cancel trade2

  8. if trade2 active:
       same trailing SL logic as trade1
```

---

## Background — 4H Market Bias (always running)

Recalculated on every closed 4H candle. Not a trade trigger — a direction filter.
Uses identical fractal + state machine logic as Phase 1, applied to 4H candles.

```python
# bull:    4H confirmed swing sequence = HH + HL
# bear:    4H confirmed swing sequence = LH + LL
# neutral: not enough confirmed swings yet — no trading

market_bias = calc_structure(candles_4h, L=5, R=5)
# returns: "bull" | "bear" | "neutral"
```

No trade fires if market_bias == "neutral".

---

## Phase 1 — BOS / CHoCH Detection

### Step 1 — Swing Point Detection (Williams Fractal)

Applied to closed 15m candles. L=5, R=5.

```python
# Swing High at index i:
is_swing_high = (
    candles[i].high > max(c.high for c in candles[i-L : i])
    and
    candles[i].high > max(c.high for c in candles[i+1 : i+R+1])
)

# Swing Low at index i:
is_swing_low = (
    candles[i].low < min(c.low for c in candles[i-L : i])
    and
    candles[i].low < min(c.low for c in candles[i+1 : i+R+1])
)
```

**Confirmation lag:** swing at index i cannot be confirmed until R=5 candles have closed after it.
On 15m this means a **75-minute lag** per swing. Unavoidable and correct.
Never detect swings on unconfirmed candles.

**Initialization:** bot needs at least L+R+1 = 11 closed 15m candles before any swing can be confirmed.
No trading until initialization is complete.

### Step 2 — Structural Node State Machine

Four variables maintained in memory, updated on every new confirmed swing:

```python
state = {
    "trend":       None,   # "bull" | "bear" | None (unknown until first CHoCH)
    "HH":          None,   # current highest confirmed swing high
    "LL":          None,   # current lowest confirmed swing low
    "strong_low":  None,   # swing low that preceded the current HH
    "strong_high": None,   # swing high that preceded the current LL
}
```

Updating rules:

```
In BULLISH trend:
  new confirmed SH > current HH:
    → HH = new SH
    → strong_low = most recent confirmed SL before this new SH

In BEARISH trend:
  new confirmed SL < current LL:
    → LL = new SL
    → strong_high = most recent confirmed SH before this new SL
```

### Step 3 — BOS and CHoCH Calculation

**Rule: break is only valid on candle CLOSE past the structural node.**
Wick cross alone = NOT a BOS/CHoCH, but DOES trigger FVG scan (see Phase 3 Gate 1).

```
BULLISH state:
  BOS:   Close_i > HH
           → trend stays bullish, begin new HH search
           → strong_low = SL prior to this break
           → fire Trade 1 (long)

  CHoCH: Close_i < strong_low
           → trend flips to BEARISH
           → strong_high = highest point reached before break
           → fire Trade 1 (short)

BEARISH state:
  BOS:   Close_i < LL
           → trend stays bearish, begin new LL search
           → strong_high = SH prior to this break
           → fire Trade 1 (short)

  CHoCH: Close_i > strong_high
           → trend flips to BULLISH
           → strong_low = lowest point reached before break
           → fire Trade 1 (long)
```

**Direction filter:** BOS/CHoCH direction must align with 4H market_bias. Counter-trend signals ignored.

---

## Phase 2 — Trade 1 (scout entry)

Fires **immediately** on BOS/CHoCH confirmation. No delay beyond bias alignment check.

### Entry

```python
entry = market_order_now()
lot   = base_lot              # e.g. 0.01
sl    = calculate_trade1_sl() # defined below
tp    = None                  # no fixed TP — trailing SL only
```

### Trade 1 SL Placement — DEFINED

SL is placed just beyond the **lowest wick extreme** reached during the liquidity sweep
that preceded the BOS. This is the exact point where the setup thesis is invalidated.

```python
# For a LONG trade (bullish BOS):
sweep_zone_candles = candles_15m[strong_low_index : bos_candle_index + 1]
sweep_candle = min(sweep_zone_candles, key=lambda c: c.low)
sl = sweep_candle.low - T1_BUFFER_PIPS

# For a SHORT trade (bearish BOS):
sweep_zone_candles = candles_15m[strong_high_index : bos_candle_index + 1]
sweep_candle = max(sweep_zone_candles, key=lambda c: c.high)
sl = sweep_candle.high + T1_BUFFER_PIPS
```

Handles multiple sweep candles automatically — `min()`/`max()` always finds the true extreme
regardless of how many candles the sweep takes.

**T1_BUFFER_PIPS:** `[BACKTEST]` — suggested: 3–5 pips on gold.

### Risk Distance

Calculated once at Trade 1 entry and stored. Used for all downstream calculations.

```python
risk_distance = abs(entry - sl)
# stored in trade1.risk_distance
```

### Point 1 — Move SL to Breakeven `[BACKTEST]`

```python
point1 = entry + risk_distance * P1_RATIO   # for longs
point1 = entry - risk_distance * P1_RATIO   # for shorts

if price crosses point1:
    sl = entry
    trade1.point1_hit = True
```

**P1_RATIO:** `[BACKTEST]` — suggested: 0.5 (price moved half the risk distance in favour)

### Point 2 — Activate Trailing SL `[BACKTEST]`

```python
point2 = entry + risk_distance * P2_RATIO   # for longs
point2 = entry - risk_distance * P2_RATIO   # for shorts

if price crosses point2:
    activate_trailing_sl()
    trade1.point2_hit = True
```

**P2_RATIO:** `[BACKTEST]` — suggested: 1.0 (price moved full risk distance in profit)

### Trailing SL Mechanic — DEFINED

Fractal-based on confirmed 1m swing points.

```python
# For longs: after each new confirmed 1m swing low,
# move SL to just below that swing low

# Uses same Williams Fractal logic as 15m but smaller window
# L=R=3 on 1m (faster response than L=R=5)

if new_1m_swing_low_confirmed:
    candidate_sl = new_1m_swing_low.low - TRAIL_BUFFER_PIPS
    if candidate_sl > current_sl:      # only move SL toward profit, never widen
        sl = candidate_sl

# For shorts: mirror logic using 1m swing highs
```

**TRAIL_BUFFER_PIPS:** `[BACKTEST]` — suggested: 2–3 pips
**1m fractal window L=R:** `[BACKTEST]` — suggested: 3

### If Trade 1 Hits SL → proceed to Phase 3

---

## Phase 3 — Trade 2 Qualification

Only reached if Trade 1 SL was hit. Two layers: hard gates then confidence scoring.
All hard gates must pass. A high confidence score does NOT override a failed gate.

---

### Layer 1 — Hard Gates

#### Gate 1 — 1m FVG exists and is clean

```python
for i in range(1, len(candles_1m) - 1):
    fvg_low  = candles_1m[i-1].high
    fvg_high = candles_1m[i+1].low
    if fvg_high > fvg_low:
        fvg = FVG(low=fvg_low, high=fvg_high, impulse_candle=candles_1m[i])

if not fvg:
    abort()

if fvg.is_refilled:
    abort()
```

FVG must be formed by the impulse candle that caused (or directly followed) the BOS.
Also applies to wick-only cross scenarios from Phase 1.

#### Gate 2 — FVG size above noise threshold

```python
atr20_1m = calculate_atr(candles_1m, period=20)

min_fvg_size = max(
    atr20_1m * FVG_ATR_MULTIPLIER,
    ABSOLUTE_MIN_PIPS
)

if (fvg.high - fvg.low) < min_fvg_size:
    abort()
```

**FVG_ATR_MULTIPLIER:** `[BACKTEST]` — suggested: 0.10–0.20
**ABSOLUTE_MIN_PIPS:** `[BACKTEST]` — suggested: 3 pips

#### Gate 3 — 15m structure still intact (time-conditional)

```python
time_since_bos = now - trade1.bos_time
SWING_LAG_MS = 75 * 60 * 1000    # 75 min = R=5 × 15m

if time_since_bos >= SWING_LAG_MS:
    if not is_15m_structure_intact(direction=bos_direction):
        abort()
# if < 75 min: skip — no new structural information available
```

#### Gate 4 — Liquidity sweep confirmed — DEFINED

**Purpose:** confirms that the wick which caused the BOS deliberately targeted a known
resting order level and closed back through it. Separates institutional traps from
random noise. The tier of the level swept also feeds into the multiplier score.

**What counts as a sweep:**
```
1. A candle wick extends THROUGH a known liquidity level
2. The same candle (or the next candle) CLOSES back on the other side
   = price grabbed the liquidity and rejected
```

**Liquidity level tiers (priority order):**

| Tier | Level | Significance |
|---|---|---|
| 1 | Previous day high / previous day low | Highest — watched by all institutional desks |
| 2 | Current session high/low, previous session high/low | Medium — session-specific participants |
| 3 | StrongHigh / StrongLow from 15m state machine | Lowest — retail stop clusters on chart |
| 3 | Equal highs / equal lows (within tolerance) | Lowest — obvious double top/bottom clusters |

**Session boundaries:**
```
Asian:  00:00 – 09:00 UTC
London: 08:00 – 17:00 UTC
NY:     13:00 – 22:00 UTC
```

**Implementation:**

```python
@dataclass
class Level:
    price: float
    tier:  int     # 1, 2, or 3

def build_liquidity_levels(session_levels, state_15m) -> list[Level]:
    levels = []

    # Tier 1 — daily
    levels.append(Level(price=session_levels.prev_day_high, tier=1))
    levels.append(Level(price=session_levels.prev_day_low,  tier=1))

    # Tier 2 — session
    levels.append(Level(price=session_levels.current_session_high, tier=2))
    levels.append(Level(price=session_levels.current_session_low,  tier=2))
    levels.append(Level(price=session_levels.prev_session_high,    tier=2))
    levels.append(Level(price=session_levels.prev_session_low,     tier=2))

    # Tier 3 — structural
    levels.append(Level(price=state_15m.strong_low,  tier=3))
    levels.append(Level(price=state_15m.strong_high, tier=3))

    # Tier 3 — equal highs/lows [RESEARCH]
    # scan confirmed 15m swing highs for clusters within EQUAL_HL_TOLERANCE pips
    # scan confirmed 15m swing lows for clusters within EQUAL_HL_TOLERANCE pips

    return levels


def is_liquidity_sweep(sweep_zone_candles, direction, levels) -> tuple[bool, int]:
    # sort by tier so tier 1 is checked first — return highest tier hit
    for level in sorted(levels, key=lambda l: l.tier):
        for candle in sweep_zone_candles:

            if direction == "bull":
                wick_through = candle.low <= level.price + SWEEP_TOLERANCE_PIPS
                close_above  = candle.close > level.price
                if wick_through and close_above:
                    return True, level.tier

            if direction == "bear":
                wick_through = candle.high >= level.price - SWEEP_TOLERANCE_PIPS
                close_below  = candle.close < level.price
                if wick_through and close_below:
                    return True, level.tier

    return False, None


# In Phase 3:
swept, sweep_tier = is_liquidity_sweep(
    sweep_zone_candles, bos_direction, build_liquidity_levels(session_levels, state_15m)
)
if not swept:
    abort()

trade1.sweep_tier = sweep_tier   # stored for use in scoring below
```

**Items still needed for Gate 4:**

| Item | Status |
|---|---|
| `SWEEP_TOLERANCE_PIPS` | `[BACKTEST]` — suggested: 3–5 pips |
| Equal highs/lows detection | `[RESEARCH]` — needs tolerance window and minimum cluster size |
| How many previous sessions to include | `[BACKTEST]` — suggested: current + 1 previous session |

---

### Layer 2 — Confidence Scoring

Only reached if all four hard gates pass.

#### Sweep Strength Score (max 5)

| Condition | Points | How determined |
|---|---|---|
| Swept a Tier 1 level (daily high/low) | +2 | sweep_tier == 1 |
| Swept a Tier 2 level (session high/low) | +1 | sweep_tier == 2 |
| Sweep wick is >50% of the candle body | +1 | sweep_candle wick/body ratio |
| Sweep occurred during London or NY session | +1 | session at time of sweep |
| Signal type is CHoCH (trend flip, not BOS) | +1 | bos_type == "choch" |

Score capped at 5.

Note: Tier 1 and Tier 2 points are mutually exclusive (only one tier is returned).
Tier 3 structural sweep passes the gate but adds 0 sweep score points.

#### FVG Quality Score (max 5)

| Condition | Points |
|---|---|
| FVG width > 15 pips on gold | +2 |
| FVG formed on the impulse candle itself (not a candle after) | +1 |
| FVG sits inside a 4H order block zone | +1 |
| FVG unfilled for >3 closed 1m candles before entry | +1 |

#### Multiplier Calculation

```python
total_score = min(sweep_score, 5) + min(fvg_score, 5)   # range: 0–10

if total_score < MIN_SCORE:
    abort()

re_entry_multiplier = (total_score / 10) * 5             # range: 0.0–5.0
lot = base_lot * max(1, floor(re_entry_multiplier))
```

**MIN_SCORE:** `[BACKTEST]` — suggested: 4 out of 10

Score to lot size (base_lot = 0.01):

| Score | Multiplier | Lot |
|---|---|---|
| 4 | 2.0 | 0.02 |
| 6 | 3.0 | 0.03 |
| 8 | 4.0 | 0.04 |
| 10 | 5.0 | 0.05 |

---

## Phase 4 — Trade 2 Entry (sniper)

### Entry

```python
entry = (fvg.low + fvg.high) / 2
sl    = fvg.low  - T2_BUFFER_PIPS   # long
      = fvg.high + T2_BUFFER_PIPS   # short
lot   = base_lot * max(1, floor(re_entry_multiplier))
tp    = None
```

**T2_BUFFER_PIPS:** `[BACKTEST]` — suggested: 2–3 pips.

### Order Validity — Cancel if any trigger before fill

```python
if price_refills_fvg():        cancel_order()
if time_elapsed > T2_TIMEOUT:  cancel_order()
if market_bias_flips():        cancel_order()
```

**T2_TIMEOUT:** `[BACKTEST]` — suggested: 10 minutes.

### After Fill — Trailing SL

Identical to Trade 1 trailing logic.
Risk distance recalculated from T2 entry to T2 SL.

---

## All Price Scenarios

| # | What price does | T1 result | T2 result |
|---|---|---|---|
| 1 | BOS → straight run, no SL hit | trails to big win | never triggered |
| 2 | BOS → Point 1 (BE) → reverses → stopped | breakeven | never triggered |
| 3 | BOS → Point 1 → Point 2 → reverses | small locked win | never triggered |
| 4 | BOS → Point 1 → Point 2 → full trail run | maximum win | never triggered |
| 5 | BOS → T1 SL → FVG → trend resumes | small loss | big win |
| 6 | BOS → T1 SL → FVG → T2 SL also hit | small loss | loss (worst case) |
| 7 | BOS → T1 SL → no FVG forms | small loss | blocked (no FVG) |
| 8 | BOS → T1 SL → FVG → hard gate fails | small loss | blocked correctly |
| 9 | BOS → T1 SL → FVG → score < MIN_SCORE | small loss | blocked (low confidence) |
| 10 | T2 limit placed → FVG refills before fill | small loss | cancelled |

**Worst case (scenario 6):** both trades lose.
Position sizing rule: T1 + T2 combined max loss must not exceed 1% of account.

---

## Drawbacks and Risks

### Conceptual risks
- **False BOS:** body-close confirmation reduces but does not eliminate fakeouts on gold.
- **Double loss (scenario 6):** contained only by position sizing, not logic.
- **FVG refills fast:** 1m FVGs can be consumed within 1–3 minutes on gold.
- **Fake FVGs:** Gate 1 (impulse candle origin) + Gate 2 (ATR filter) are the defences.
- **Tier 3 sweep only:** passes the gate but contributes 0 to sweep score → low multiplier → small lot. This is correct behaviour — low conviction setup gets small size.

### Execution risks
- **Spread during news:** NFP, CPI, FOMC can produce 30–50 pip spreads. `[RESEARCH]` news filter needed.
- **T1 slippage:** risk_distance uses actual fill price so slippage is absorbed automatically, but effective SL distance shrinks.
- **Partial T2 fill:** `[RESEARCH]` define minimum fill threshold before activating trade management.

---

## Full Items Status

### Defined and locked

| Item | Definition |
|---|---|
| BOS/CHoCH detection | Williams Fractal L=R=5, close-only confirmation, state machine |
| 4H bias calculation | same fractal logic on 4H candles |
| Trade 1 SL placement | lowest/highest wick extreme across sweep zone candles, ± buffer |
| FVG 3-candle gap rule | candle[i-1].high to candle[i+1].low |
| FVG ATR-relative size filter | atr20 × multiplier + absolute pip floor |
| 15m structure gate (time-conditional) | skip if < 75 min since BOS, run if ≥ 75 min |
| is_liquidity_sweep | wick-through + close-back on tiered levels, tier stored for scoring |
| Liquidity level tiers | Tier 1 = daily, Tier 2 = session, Tier 3 = structural/equal HL |
| Session boundaries | Asian 00–09 UTC, London 08–17 UTC, NY 13–22 UTC |
| Sweep tier → score mapping | Tier 1 = +2, Tier 2 = +1, Tier 3 = +0 |
| re_entry_multiplier formula | (sweep + fvg score) / 10 × 5, each score capped at 5 |
| T2 entry placement | limit at FVG midpoint |
| Trailing SL mechanic | fractal-based on confirmed 1m swing lows/highs, one-way only |
| No fixed TP on either trade | trail-only confirmed |

### Needs research (conceptual decision required)

| Item | What needs deciding |
|---|---|
| `[RESEARCH]` Equal highs/lows detection | tolerance window + minimum cluster size for Tier 3 |
| `[RESEARCH]` News/event filter | which events, block window duration |
| `[RESEARCH]` Partial T2 fill handling | minimum fill size before activating trade management |

### Needs backtesting (value TBD)

| Item | Suggested starting value |
|---|---|
| `[BACKTEST]` T1_BUFFER_PIPS | 3–5 pips |
| `[BACKTEST]` P1_RATIO (Point 1) | 0.5 × risk_distance |
| `[BACKTEST]` P2_RATIO (Point 2) | 1.0 × risk_distance |
| `[BACKTEST]` 1m fractal window L=R | 3 |
| `[BACKTEST]` TRAIL_BUFFER_PIPS | 2–3 pips |
| `[BACKTEST]` FVG_ATR_MULTIPLIER | 0.10–0.20 |
| `[BACKTEST]` ABSOLUTE_MIN_PIPS | 3 pips |
| `[BACKTEST]` MIN_SCORE threshold | 4 out of 10 |
| `[BACKTEST]` T2_BUFFER_PIPS | 2–3 pips |
| `[BACKTEST]` T2_TIMEOUT | 10 minutes |
| `[BACKTEST]` SWEEP_TOLERANCE_PIPS | 3–5 pips |
| `[BACKTEST]` Previous sessions to include | current + 1 previous |

---

## Development Phases

```
Phase 1 — current priority
  Candle builder from tick data (1m, 15m, 4H)
  Session level tracker (daily + session high/low)
  FVG scanner (3-candle gap + ATR size filter)
  re_entry_multiplier scoring engine
  Trailing SL logic (fractal-based on 1m, L=R=3)

Phase 2
  Williams Fractal swing detector (15m + 4H, L=R=5)
  Structural node state machine (HH / LL / StrongLow / StrongHigh)
  BOS/CHoCH detection (close-only confirmation)
  4H market_bias calculator
  Trade 1 SL calculator (sweep extreme finder)
  is_liquidity_sweep (tiered level check)

Phase 3
  Connect Phase 1 + Phase 2 into full pipeline
  Equal highs/lows detection for Tier 3 levels
  News/event filter
  Partial fill handling for T2

Phase 4
  Backtesting on historical XAUUSD tick/candle data
  Resolve all [BACKTEST] values
  Risk and drawdown analysis
  Validate: T1 + T2 combined max loss never exceeds 1% of account
```
