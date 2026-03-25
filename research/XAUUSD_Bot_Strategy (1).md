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

  # trade state — updated by bot logic
  "trade1": {
    "active":      bool,
    "entry":       float,
    "sl":          float,
    "lot":         float,
    "point1_hit":  bool,
    "point2_hit":  bool,
    "trailing":    bool,
    "bos_time":    int,    # unix ms when BOS fired — used for 15m gate timing
  },
  "trade2": {
    "pending":     bool,
    "entry":       float,  # FVG midpoint
    "sl":          float,
    "lot":         float,
    "fvg_low":     float,
    "fvg_high":    float,
    "multiplier":  float,
    "expire_at":   int,    # unix ms timeout
  }
}
```

---

## Bot State Machine (runs on every tick)

```
on every tick:
  1. update open bar (1m, 15m, 4H) with new price

  2. if 1m bar just closed:
       run FVG scanner on last 3 closed 1m candles
       run swing detection on 1m candles (for trailing SL reference)

  3. if 15m bar just closed:
       run swing detection on 15m candles (L=R=5)
       update 15m structural node state (HH, LL, StrongLow, StrongHigh)
       check for BOS/CHoCH on 15m

  4. if 4H bar just closed:
       recalculate market_bias from 4H structure

  5. if trade1 active:
       check if price hit point1  → move SL to breakeven
       check if price hit point2  → activate trailing SL
       check if trailing active   → update SL to last confirmed structure point
       check if price hit SL      → close trade1, trigger Phase 3 evaluation

  6. if trade2 pending:
       check if price reached FVG midpoint → open trade2
       check if FVG is still valid (not refilled)
       check if timeout expired   → cancel trade2
       check if market_bias flipped → cancel trade2

  7. if trade2 active:
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

**Confirmation lag:** a swing at index i cannot be confirmed until R=5 candles have closed after it.
On 15m this means a **75-minute lag** per swing. This is unavoidable and correct.
Never detect swings on unconfirmed (partially closed) candles.

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

**Rule: break is only valid on candle CLOSE past the node.**
Wick cross alone = NOT a BOS/CHoCH, but DOES trigger FVG scan (see Phase 3 Gate 1).

```
BULLISH state:
  BOS:   Close_i > HH
           → trend stays bullish, begin new HH search
           → strong_low = SL prior to this break
           → fire Trade 1

  CHoCH: Close_i < strong_low
           → trend flips to BEARISH
           → strong_high = highest point reached before break
           → fire Trade 1 (short)

BEARISH state:
  BOS:   Close_i < LL
           → trend stays bearish, begin new LL search
           → strong_high = SH prior to this break
           → fire Trade 1

  CHoCH: Close_i > strong_high
           → trend flips to BULLISH
           → strong_low = lowest point reached before break
           → fire Trade 1 (long)
```

**Direction filter:** BOS/CHoCH direction must align with 4H market_bias. Counter-trend signals are ignored.

---

## Phase 2 — Trade 1 (scout entry)

Fires **immediately** on BOS/CHoCH confirmation. No delay beyond bias alignment check.

### Entry

```python
entry = market_order_now()
lot   = base_lot              # e.g. 0.01
sl    = trade1_sl()           # see below
tp    = None                  # no fixed TP — trailing SL only
```

### Trade 1 SL Placement `[RESEARCH]`

SL placed just beyond the liquidity sweep point that caused the BOS/CHoCH.
Three candidate options — needs final decision:

```
Option A — StrongLow / StrongHigh from state machine (recommended)
  sl = strong_low - buffer_pips   (for longs)
  sl = strong_high + buffer_pips  (for shorts)
  pros: most structurally meaningful, directly tied to BOS origin
  cons: may be far if the swing was large (wide SL = smaller lot for same risk)

Option B — low/high of the BOS candle itself
  sl = bos_candle.low - buffer_pips   (for longs)
  pros: tighter SL
  cons: less structurally grounded, higher false stop-out rate

Option C — most recent confirmed 15m swing low/high
  sl = last_confirmed_sl - buffer_pips   (for longs)
  pros: intermediate between A and B
  cons: may differ from StrongLow depending on recent structure
```

**Recommended starting point:** Option A.

**buffer_pips:** `[BACKTEST]` — suggested: 3–5 pips on gold to absorb spread and wick noise.

### Point 1 — Move SL to Breakeven `[BACKTEST]`

```python
risk_distance = abs(entry - sl)
point1 = entry + risk_distance * P1_RATIO   # for longs

if price >= point1:
    sl = entry    # trade is now risk-free
```

**P1_RATIO:** `[BACKTEST]` — suggested starting value: 0.5 (half the original risk distance)

### Point 2 — Activate Trailing SL `[BACKTEST]`

```python
point2 = entry + risk_distance * P2_RATIO   # for longs

if price >= point2:
    activate_trailing_sl()
    # from here: no fixed exit, trail runs until structure breaks
```

**P2_RATIO:** `[BACKTEST]` — suggested starting value: 1.0 (price moved full risk distance in profit)

### Trailing SL Mechanic `[RESEARCH]`

Three options — needs final decision:

```
Option A — fractal-based on 1m (recommended):
  after each new confirmed 1m swing low forms (for longs),
  move SL to just below that swing low
  pros: structurally grounded, adapts naturally to momentum
  cons: 5-candle confirmation lag on 1m = ~5 min delay on SL updates

Option B — previous closed 1m candle low:
  on each closed 1m candle, SL = that candle's low
  pros: simple, no lag
  cons: too tight on volatile gold candles, frequent premature exits

Option C — ATR-based:
  SL = current_price - (ATR20_1m × TRAIL_ATR_MULTIPLIER)
  pros: volatility-adaptive
  cons: not structurally grounded, may trail too loosely
```

**Recommended:** Option A. Same trailing logic applies identically to Trade 2 once filled.

### If Trade 1 Hits SL → proceed to Phase 3

---

## Phase 3 — Trade 2 Qualification

Only reached if Trade 1 SL was hit. Two layers: hard gates then confidence scoring.
All hard gates must pass. A high confidence score does NOT override a failed gate.

---

### Layer 1 — Hard Gates

#### Gate 1 — 1m FVG exists and is clean

```python
# 3-candle gap rule on closed 1m candles:
for i in range(1, len(candles_1m) - 1):
    fvg_low  = candles_1m[i-1].high
    fvg_high = candles_1m[i+1].low
    if fvg_high > fvg_low:
        fvg = FVG(low=fvg_low, high=fvg_high, impulse_candle=candles_1m[i])

if not fvg:
    abort()

if fvg.is_refilled:    # any subsequent candle has traded through the gap
    abort()
```

FVG must be formed by the impulse candle that caused (or directly followed) the BOS.
Random gaps elsewhere on the chart do not qualify.

Also applies when Phase 1 detected a wick-only cross (not a full BOS):
scan for FVG formed by that wick candle's impulse move.

#### Gate 2 — FVG size above noise threshold

```python
atr20_1m = calculate_atr(candles_1m, period=20)

min_fvg_size = max(
    atr20_1m * FVG_ATR_MULTIPLIER,   # volatility-relative minimum
    ABSOLUTE_MIN_PIPS                 # hard floor regardless of ATR
)

if (fvg.high - fvg.low) < min_fvg_size:
    abort()
```

**FVG_ATR_MULTIPLIER:** `[BACKTEST]` — suggested range: 0.10–0.20

**ABSOLUTE_MIN_PIPS:** `[BACKTEST]` — suggested: 3 pips on gold

Rationale: ATR-relative scaling means the filter tightens during high-volatility sessions
(London open) and relaxes proportionally during low-volatility sessions (Asian).

#### Gate 3 — 15m structure still intact (time-conditional)

```python
time_since_bos = now - trade1.bos_time
SWING_LAG_MS = 75 * 60 * 1000    # 75 min = R=5 candles × 15m

if time_since_bos >= SWING_LAG_MS:
    # new 15m swings may have formed since BOS fired
    # check if structure has broken against the original BOS direction
    if not is_15m_structure_intact(direction=bos_direction):
        abort()

# if time_since_bos < 75 min:
#   skip — swing history is identical to when BOS fired
#   no new structural information exists yet
#   scoring gates below are the protection in this window
```

#### Gate 4 — Liquidity sweep confirmed `[RESEARCH]`

Price must have swept a known liquidity level before reversing to form the FVG.

```python
def is_liquidity_sweep(direction) -> bool:
    # Check in priority order:
    # 1. Daily high / daily low
    # 2. Current session high / low (London, NY, Asian)
    # 3. Recent swing high / low from 15m state machine (StrongHigh / StrongLow)
    pass

# [RESEARCH] exact lookback period for each level type
# [RESEARCH] pip tolerance — how close to level counts as "swept"
# [RESEARCH] whether all three tiers are checked or first hit suffices
```

---

### Layer 2 — Confidence Scoring

Only reached if all four hard gates pass.

#### Sweep Strength Score (max 5)

| Condition | Points |
|---|---|
| Swept a 4H or Daily structural high/low | +2 |
| Sweep wick is >50% of the candle body | +1 |
| Sweep occurred during London or NY session | +1 |
| Swept equal highs / equal lows (obvious liquidity pool) | +1 |
| Signal type is CHoCH (trend flip, not just BOS) | +1 |

Score is capped at 5 even if conditions sum to 6.

#### FVG Quality Score (max 5)

| Condition | Points |
|---|---|
| FVG width > 15 pips on gold | +2 |
| FVG formed on the impulse candle itself (not a candle after) | +1 |
| FVG sits inside a 4H order block zone | +1 |
| FVG unfilled for >3 closed 1m candles before entry | +1 |

#### Multiplier Calculation

```python
total_score = sweep_score + fvg_score           # range: 0–10

if total_score < MIN_SCORE:
    abort()    # not confident enough for Trade 2

re_entry_multiplier = (total_score / 10) * 5    # range: 0.0–5.0
lot = base_lot * max(1, floor(re_entry_multiplier))
```

**MIN_SCORE:** `[BACKTEST]` — suggested starting value: 4 out of 10

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
entry = (fvg.low + fvg.high) / 2               # limit order at FVG midpoint
sl    = fvg.low - T2_BUFFER_PIPS               # just below FVG bottom (long)
      = fvg.high + T2_BUFFER_PIPS              # just above FVG top (short)
lot   = base_lot * max(1, floor(re_entry_multiplier))
tp    = None                                    # trail only — same as Trade 1
```

**T2_BUFFER_PIPS:** `[BACKTEST]` — suggested: 2–3 pips. Must account for gold spread.

### Order Validity — Cancel if any trigger before fill

```python
if price_refills_fvg():        cancel_order()   # gap traded through before fill
if time_elapsed > T2_TIMEOUT:  cancel_order()   # order timed out
if market_bias_flips():        cancel_order()   # 4H structure changed direction
```

**T2_TIMEOUT:** `[BACKTEST]` — suggested: 10 minutes. 1m FVGs on gold invalidate fast.

### After Fill — Trailing SL

Identical to Trade 1 trailing logic (same Point 1 / Point 2 ratios, same fractal trail mechanic).
Risk distance R is calculated from T2 entry to T2 SL.

---

## All Price Scenarios

| # | What price does | T1 result | T2 result |
|---|---|---|---|
| 1 | BOS → straight run, no SL hit | trails to big win | never triggered |
| 2 | BOS → hits Point 1 (BE) → reverses → stopped | breakeven | never triggered |
| 3 | BOS → Point 1 → Point 2 → reverses | small locked win | never triggered |
| 4 | BOS → Point 1 → Point 2 → full trail run | maximum win | never triggered |
| 5 | BOS → T1 SL → FVG found → trend resumes | small loss | big win |
| 6 | BOS → T1 SL → FVG → T2 SL also hit | small loss | loss (worst case) |
| 7 | BOS → T1 SL → no FVG forms | small loss | blocked (no FVG) |
| 8 | BOS → T1 SL → FVG → hard gate fails | small loss | blocked correctly |
| 9 | BOS → T1 SL → FVG → score < MIN_SCORE | small loss | blocked (low confidence) |
| 10 | T2 limit placed → FVG refills before fill | small loss | cancelled |

**Worst case (scenario 6):** both T1 and T2 lose.
Position sizing rule: combined max loss of T1 + T2 at full multiplier must not exceed 1% of account.

---

## Drawbacks and Risks

### Conceptual risks
- **False BOS:** gold wicks aggressively through structural levels. Body-close confirmation reduces but does not eliminate fakeouts.
- **Double loss (scenario 6):** cannot be prevented by logic alone — must be contained by position sizing.
- **FVG refills fast on gold:** 1m FVGs can be consumed within 1–3 minutes during high volatility. T2_TIMEOUT must be tight.
- **Fake FVGs during consolidation:** Gate 1 (impulse candle origin) and Gate 2 (ATR size filter) are the primary defences.
- **CHoCH bonus in scoring:** capping sweep_score at 5 prevents over-sizing when CHoCH adds the extra point.

### Execution risks
- **Spread during news:** NFP, CPI, FOMC can widen gold spread to 30–50 pips. T2 tight SL gets hit by spread alone. `[RESEARCH]` news filter needed.
- **T1 slippage:** BOS candles move fast. T1 market order may fill 10–15 pips late. SL distance should account for this.
- **Partial T2 fill:** limit order at FVG midpoint may fill partially if price only briefly touches the zone. `[RESEARCH]` define minimum fill threshold.

---

## Full Items Status

### Defined and locked

| Item | Definition |
|---|---|
| BOS/CHoCH detection | Williams Fractal L=R=5, close-only confirmation, state machine |
| 4H bias calculation | same fractal logic on 4H candles |
| FVG 3-candle gap rule | candle[i-1].high to candle[i+1].low |
| FVG ATR-relative size filter | atr20 × multiplier + absolute pip floor |
| 15m structure gate (time-conditional) | skip if < 75 min since BOS, run if ≥ 75 min |
| re_entry_multiplier formula | (sweep + fvg score) / 10 × 5, capped scores |
| T2 entry placement | limit at FVG midpoint |
| Trailing SL mechanic (method chosen) | fractal-based on 1m confirmed swings |
| No fixed TP on either trade | trail-only confirmed |

### Needs research (conceptual decision required)

| Item | What needs deciding |
|---|---|
| `[RESEARCH]` Trade 1 SL placement | Option A (StrongLow) vs B (BOS candle low) vs C (recent swing low) |
| `[RESEARCH]` is_liquidity_sweep definition | lookback period, pip tolerance, level priority |
| `[RESEARCH]` News/event filter | which events, block window duration |
| `[RESEARCH]` Partial T2 fill handling | minimum fill size before activating trade management |

### Needs backtesting (value TBD)

| Item | Suggested starting value |
|---|---|
| `[BACKTEST]` T1 SL buffer pips | 3–5 pips beyond StrongLow/StrongHigh |
| `[BACKTEST]` P1_RATIO (Point 1) | 0.5 × risk distance |
| `[BACKTEST]` P2_RATIO (Point 2) | 1.0 × risk distance |
| `[BACKTEST]` FVG_ATR_MULTIPLIER | 0.10–0.20 |
| `[BACKTEST]` ABSOLUTE_MIN_PIPS | 3 pips |
| `[BACKTEST]` MIN_SCORE threshold | 4 out of 10 |
| `[BACKTEST]` T2_BUFFER_PIPS | 2–3 pips |
| `[BACKTEST]` T2_TIMEOUT | 10 minutes |

---

## Development Phases

```
Phase 1 — current priority
  Candle builder from tick data (1m, 15m, 4H)
  FVG scanner (3-candle gap + ATR size filter)
  re_entry_multiplier scoring engine
  Trailing SL logic (fractal-based on 1m)

Phase 2
  Williams Fractal swing detector (15m + 4H)
  Structural node state machine (HH / LL / StrongLow / StrongHigh)
  BOS/CHoCH detection (close-only confirmation)
  4H market_bias calculator

Phase 3
  Connect Phase 1 + Phase 2 into full pipeline
  is_liquidity_sweep implementation
  News/event filter
  Resolve Trade 1 SL placement [RESEARCH]

Phase 4
  Backtesting on historical XAUUSD tick/candle data
  Resolve all [BACKTEST] values
  Risk and drawdown analysis
  Validate: T1 + T2 combined max loss never exceeds 1% of account
```
