# XAUUSD Liquidity Sweep Bot — Full Strategy Document

---

## Overview

A two-trade pipeline that exploits liquidity sweeps and market structure breaks on gold (XAUUSD). Trade 1 is a small scout entry on BOS/CHoCH confirmation. Trade 2 is a larger sniper entry using FVG precision if Trade 1 is stopped out. No fixed TP on either trade — trailing SL only.

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

  # candle state — bot builds from ticks, not from data rows
  "bar_1m":  {
    "open": float, "high": float, "low": float,
    "close": float, "closed": bool
  },
  "bar_15m": {
    "open": float, "high": float, "low": float,
    "close": float, "closed": bool
  },

  # trade state — updated by bot logic
  "trade1": {
    "active":      bool,
    "entry":       float,
    "sl":          float,
    "lot":         float,
    "point1_hit":  bool,
    "point2_hit":  bool,
    "trailing":    bool,
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
  1. update open bar (1m, 15m) with new price
  2. if bar just closed:
       run FVG scanner on closed candles
       recalculate 15m structure if 15m bar closed
       recalculate 4H structure if 4H bar closed
  3. if trade1 active:
       check if price hit point1  → move SL to breakeven
       check if price hit point2  → activate trailing SL
       check if trailing active   → update SL to last structure low/high
       check if price hit SL      → close trade1, trigger Phase 3 evaluation
  4. if trade2 pending:
       check if price reached FVG midpoint → open trade2
       check if FVG is still valid (not refilled)
       check if timeout expired   → cancel trade2
       check if market_bias flipped → cancel trade2
  5. if trade2 active:
       same trailing SL logic as trade1
```

---

## Background — 4H Market Bias (always running)

Recalculated on every closed 4H candle. Not a trade trigger — a direction filter used by all phases.

```python
def calc_4h_structure(candles_4h) -> str:
    # bull: price making higher highs AND higher lows
    # bear: price making lower highs AND lower lows
    # returns: "bull" | "bear" | "neutral"
```

Stored as `market_bias` and referenced by all downstream phases.

---

## Phase 1 — BOS / CHoCH Detection (trigger)

Checked on every closed 1m or 15m candle.

```
Condition:
  candle closes BEYOND the previous structural high (for bull BOS)
  candle closes BEYOND the previous structural low  (for bear BOS)
  AND direction aligns with market_bias

Result:
  confirmed → fire Phase 2 immediately
  not confirmed → keep waiting
```

**Notes:**
- Body close vs wick close rule: TBD (parked — needs research)
- Exact swing point identification logic: TBD (parked — core research task)
- False BOS filter: TBD (gold wicks aggressively through levels)

---

## Phase 2 — Trade 1 (scout entry)

Fires immediately on BOS/CHoCH confirmation. No delay, no extra filter.

### Entry

```python
entry  = market_order_now()
lot    = base_lot              # e.g. 0.01
sl     = liquidity_sweep_point # the high/low that was swept to cause the BOS
tp     = None                  # no fixed TP — trailing SL only
```

### Price goes in favour — SL management

```
Price hits Point 1:
  → move SL to entry (breakeven)
  → trade is now risk-free

Price hits Point 2:
  → activate trailing SL
  → trail SL behind each new confirmed structure point (not pip-based)
  → no fixed exit — let trail run until structure breaks naturally
```

**Point 1 and Point 2 exact values: TBD — needs backtesting on gold**

Suggested starting values to test:
- Point 1 = 0.5R (half the original risk distance)
- Point 2 = 1.0R (full risk distance in profit)

### Price hits SL

Trade 1 closes at a small loss. The SL is at the liquidity sweep point — the level that was swept to create the BOS. Price returning here means the sweep may have been deeper than expected. Proceed to Phase 3.

---

## Phase 3 — Trade 2 Qualification

Only reached if Trade 1 SL was hit. Two layers: hard gates then confidence scoring.

---

### Layer 1 — Hard Gates (all must pass or abort)

All four must pass. A high confidence score does NOT override a failed hard gate.

#### Gate 1 — 1m FVG exists and is clean

```python
fvg = scan_1m_fvg()

# 3-candle gap rule:
for i in range(1, len(candles_1m) - 1):
    fvg_low  = candles_1m[i-1].high
    fvg_high = candles_1m[i+1].low
    if fvg_high > fvg_low:
        fvg = FVG(low=fvg_low, high=fvg_high)

# abort conditions:
if not fvg:
    abort()  # no gap found
if fvg.is_refilled:
    abort()  # price already traded through the gap
```

FVG must be formed by the impulse candle that caused (or followed) the BOS — not a random gap elsewhere on the chart.

#### Gate 2 — FVG gap size above noise threshold

```python
if fvg.size_pips < MIN_FVG_PIPS:
    abort()
```

**MIN_FVG_PIPS: TBD — needs backtesting. Suggested starting value: 5 pips on gold.**
Rationale: gaps smaller than spread + commission are untradeable and likely noise.

#### Gate 3 — 15m structure still intact

```python
if not is_15m_structure_intact(direction=bos_direction):
    abort()
```

Checks that 15m is still making HH/HL (for bull) or LH/LL (for bear) in the same direction as the original BOS. If 15m structure has broken against the trade direction, no FVG entry is valid regardless of score. This is a hard gate — NOT part of the multiplier scoring.

#### Gate 4 — Liquidity sweep confirmed

```python
if not is_liquidity_sweep():
    abort()
```

Price must have swept a known liquidity level before reversing to form the FVG. Sweep sources in priority order:
- Daily high / daily low
- Current session high / session low (London, NY, Asian)
- Swing high / swing low from recent structure

**Exact definition and lookback period: TBD — parked for research.**

---

### Layer 2 — Confidence Scoring (soft filter)

Only reached if all four hard gates pass.

#### Sweep Strength Score (0–5)

| Condition | Points |
|---|---|
| Swept a 4H or Daily structural high/low | +2 |
| Sweep wick is >50% of the candle body | +1 |
| Sweep occurred during London or NY session | +1 |
| Swept equal highs / equal lows (obvious liquidity pool) | +1 |

#### FVG Quality Score (0–5)

| Condition | Points |
|---|---|
| FVG width > 15 pips on gold | +2 |
| FVG formed on the impulse candle itself (not after) | +1 |
| FVG sits inside a 4H order block zone | +1 |
| FVG is unfilled for >3 closed 1m candles before entry | +1 |

#### Multiplier Calculation

```python
total_score = sweep_score + fvg_score          # range: 0–10

if total_score < MIN_SCORE:
    abort()                                    # not confident enough

re_entry_multiplier = (total_score / 10) * 5  # range: 0.0–5.0
lot = base_lot * max(1, floor(re_entry_multiplier))
```

**MIN_SCORE: TBD — suggested starting value: 4 out of 10**

Score → multiplier → lot size examples (base_lot = 0.01):

| Score | Multiplier | Lot size |
|---|---|---|
| 4 | 2.0 | 0.02 |
| 6 | 3.0 | 0.03 |
| 8 | 4.0 | 0.04 |
| 10 | 5.0 | 0.05 |

---

## Phase 4 — Trade 2 Entry (sniper)

### Entry

```python
entry  = (fvg.low + fvg.high) / 2             # limit order at FVG midpoint
sl     = fvg.low - buffer_pips                 # just below FVG bottom (long)
                                               # just above FVG top (short)
lot    = base_lot * max(1, floor(re_entry_multiplier))
tp     = None                                  # no fixed TP — trail only
```

**buffer_pips: TBD — suggested 2–3 pips on gold, must account for spread**

### Order Validity — Cancel if any trigger before fill

```python
if price_refills_fvg():      cancel_order()   # FVG traded through before fill
if time_elapsed > TIMEOUT:   cancel_order()   # order expires
if market_bias_flips():      cancel_order()   # 4H structure changed direction
```

**TIMEOUT: TBD — suggested 10 minutes for 1m FVGs on gold (fills fast)**

### After fill — Trail SL

Identical to Trade 1 trail logic:
```
trail SL behind each new confirmed structure point
exit naturally when structure breaks against position
no fixed TP
```

---

## All Price Scenarios

| # | What price does | Outcome |
|---|---|---|
| 1 | BOS → straight run, no SL hit | T1 trails to big win |
| 2 | BOS → hits Point 1 (BE) → reverses → stopped | T1 breakeven · no T2 |
| 3 | BOS → Point 1 → Point 2 → reverses | T1 small locked win · no T2 |
| 4 | BOS → Point 1 → Point 2 → full trail run | T1 maximum win (best case) |
| 5 | BOS → T1 SL hit → FVG found → trend resumes | T1 small loss · T2 big win |
| 6 | BOS → T1 SL hit → FVG found → T2 SL also hit | Double loss (worst case) |
| 7 | BOS → T1 SL hit → no FVG forms | T1 loss only · sit out |
| 8 | BOS → T1 SL hit → FVG found → hard gate fails | T1 loss · T2 blocked correctly |
| 9 | BOS → T1 SL hit → FVG found → score < MIN_SCORE | T1 loss · T2 skipped (low confidence) |
| 10 | T2 limit order placed → FVG refills before fill | T1 loss · T2 cancelled |

---

## Known Drawbacks and Risks

### Conceptual risks
- **False BOS on 1m**: extremely common on gold. Wick-through without body close looks like BOS to a naive detector. Swing point logic must handle this correctly.
- **Double loss scenario (case 6)**: T1 and T2 both lose. Position sizing must ensure two simultaneous max losses stay within total account risk limit (e.g. combined ≤ 1% account).
- **FVG fills fast on gold**: 1m FVGs can be consumed within 1–3 minutes. TIMEOUT must be aggressive.
- **Fake FVGs during consolidation**: not every 3-candle gap is a real imbalance. The impulse candle rule (Gate 1) and MIN_FVG_PIPS (Gate 2) are the filters.

### Execution risks
- **Spread during news**: NFP, CPI, FOMC can produce 30–50 pip spreads on gold. T2 tight SL will get blown. Add a news/high-impact event filter.
- **Slippage on T1 market order**: BOS candles on gold move fast. T1 may fill 10–15 pips late. Factor into SL distance calculation.
- **Lot size ratio**: base_lot (T1) vs max lot (T2) = 1:5. Ensure total exposure across both open trades stays within risk rules.

### Bot-specific risks
- BOS detection is the single hardest component. All downstream logic depends on its accuracy.
- FVG scanner must re-validate that the gap is unfilled at the moment the limit order is placed, not just when it was first detected.
- 4H bias is calculated from candles, not from a data feed tag — must handle partial (unclosed) 4H candles correctly.

---

## Items Parked for Research / Backtesting

| Item | Notes |
|---|---|
| Exact swing point identification for BOS | Core research task. Affects false signal rate heavily. |
| Body close vs wick close for BOS confirmation | Gold wicks aggressively. Likely body close is safer. |
| Point 1 exact value | Suggested: 0.5R. Needs backtesting. |
| Point 2 exact value | Suggested: 1.0R. Needs backtesting. |
| MIN_FVG_PIPS | Suggested: 5 pips on gold. Needs backtesting. |
| MIN_SCORE threshold | Suggested: 4/10. Needs backtesting. |
| is_liquidity_sweep exact definition | Daily vs session vs swing. Lookback period TBD. |
| TIMEOUT for T2 FVG order | Suggested: 10 minutes. Needs testing on gold. |
| buffer_pips for SL below FVG | Suggested: 2–3 pips. Must account for spread. |
| News filter | Block trading during high-impact events. |

---

## Development Phases

```
Phase 1 (current)
  FVG scanner (1m 3-candle gap detection)
  re_entry_multiplier scoring
  Trailing SL logic (structure-based, not pip-based)
  Tick data ingestion and candle builder

Phase 2
  BOS / CHoCH detection
  Swing point identification
  4H structure calculator

Phase 3
  Connect Phase 1 + Phase 2 into full pipeline
  is_liquidity_sweep implementation
  News/event filter

Phase 4
  Backtesting on historical XAUUSD data
  Tune: Point 1, Point 2, MIN_FVG_PIPS, MIN_SCORE, TIMEOUT, buffer_pips
  Risk and drawdown analysis
```
