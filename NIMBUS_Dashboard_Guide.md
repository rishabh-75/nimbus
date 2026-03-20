# NIMBUS Dashboard Reference Guide
## Unified Mean-Reversion System

---

## THE CORE IDEA

You're looking for stocks that have pulled back hard enough to bounce. The system scores how "ripe" a pullback is across 7 dimensions, then tells you whether to enter, at what size, and when to exit.

**Entry logic:** WR(30) < -30 AND below SMA(20) AND MFI > 30
**Exit logic:** +3% profit target OR BBW contraction (above SMA + BB width shrinking) OR 30-day max hold
**Validated:** OOS Sharpe 4.17, Win 85.3%, Avg +2.44% per trade

---

## HEADER BADGES (top row)

Read left to right. Each badge is green (favorable), gold (caution), or red (unfavorable) FOR A MEAN-REVERSION ENTRY.

| Badge | Green means | Red means |
|-------|-----------|----------|
| **TIER** | PRIMARY or SECONDARY entry triggered | NO ENTRY — conditions not met |
| **WR(30)** | DEEP (<-50) or IN ZONE (<-30) — oversold | Above -30 — not oversold enough |
| **MFI** | STRONG (≥50) or OK (≥30) — money flowing in | WEAK (<30) — no buying pressure |
| **SMA** | BELOW SMA — stock is in a pullback | ABOVE SMA — not a pullback |
| **DD** | Deep drawdown from 50d high + red streak | Shallow dip — minor noise |
| **REGIME** | TREND_FRIENDLY from options | PINNING — dealers suppress moves |

**What you want to see:** All green badges = high-conviction pullback.

---

## KPI TILES (below chart)

| Tile | What it shows | Good for entry | Bad for entry |
|------|--------------|----------------|---------------|
| **SPOT** | Current price | — | — |
| **WR(30)** | Williams %R (30-day). Measures where price is within its 30-day high-low range | DEEP (<-50): extremely oversold | Above -30: not oversold |
| **MFI** | Money Flow Index (14-day). Combines price + volume to detect institutional accumulation | Strong (≥50): institutions buying the dip | Weak (<30): nobody accumulating |
| **vs SMA** | Price relative to 20-day Simple Moving Average | BELOW: pullback from trend | ABOVE: no pullback |
| **PCR OI** | Put-Call Ratio from options open interest | ≥1.3: put writers confident (floor) | ≤0.7: bearish positioning |
| **NET GEX** | Gamma Exposure. Shows whether dealers amplify or dampen moves | Negative: sharp moves possible | Positive: pinning, suppressed |
| **TO EXPIRY** | Days to nearest options expiry | ≥5d: safe window | ≤2d: pin risk high |
| **VOL STATE** | Bollinger Band width state | SQUEEZE: coiled spring | EXPANDED: move already happened |

---

## SIDEBAR (left panel)

Quick snapshot of the signal state:

- **Tier:** PRIMARY (all conditions met, FULL size), SECONDARY (core only, HALF size), or NONE
- **WR(30):** Current oversold reading. Green when < -30 (in entry zone)
- **MFI:** Money flow strength. Green ≥30, emerald ≥50
- **SMA / DD / Red:** Combined line showing distance from SMA, drawdown from 50d high, and consecutive red days
- **Score:** Base score + options overlay + filing overlay → final (0-100)
- **Breakdown:** Shows how score was computed (e.g., base=71 + opt=-5 → 66)

---

## DUAL-MODE STRIP (below filing strip)

Format: `[▶ PRIMARY] 78 [STRONG] [SIZE: FULL] ▶ PRIMARY: WR(30)=-93 | -5.0% vs SMA | MFI=50 | DD=-8.2% | 4d red | Vol=1.3x`

- **▶ PRIMARY / ▷ SECONDARY / MEAN REV**: Entry tier or no entry
- **Score + Label**: 78 STRONG, 66 GOOD, 45 WATCH, or <45 AVOID
- **SIZE**: FULL (PRIMARY tier), HALF (SECONDARY tier), SKIP (no entry)
- **Entry reason**: Full breakdown of why entry triggered (or what's missing)

---

## PANEL A: WORKFLOW ANALYSIS (bottom left)

### Regime Tile (green/gold/red border)
Shows: `MEAN REVERSION: Primary Entry` or `Secondary Entry` or `No Entry`
With detail line: all 7 indicator values at a glance.

### Trend & State Line
`Mean reversion · WR(30)=-96 · -12.0% vs SMA · MFI=24`

### Williams %R Line
Interprets WR value:
- "Deeply oversold → Strong bounce expected" (WR < -50)
- "In mean-reversion zone → Pullback entry" (WR -30 to -50)
- "Not oversold → Wait" (WR > -30)

### MFI Flow Line
Interprets money flow:
- "Strong accumulation during pullback" (MFI ≥ 50)
- "Acceptable money flow" (MFI 30-50)
- "Weak — no buying pressure in this dip" (MFI < 30)

### Verdict Box (colored border)
**Green border:** Entry signal with full reasoning and exit rules
**Gold border:** Core conditions met but missing PRIMARY qualifiers
**Red border:** No entry — lists what's missing

---

## PANEL B: TRADE VIABILITY (bottom center)

### Score (large number)
0-100 composite score. Color-coded:
- **≥75 STRONG**: High-conviction setup
- **≥60 GOOD**: Tradeable with normal sizing
- **≥45 WATCH**: Marginal — wait for improvement
- **<45 AVOID**: Don't trade

### Size Badge
- **FULL**: PRIMARY tier entry — all conditions met (OOS validated at 85% win)
- **HALF**: SECONDARY tier — core conditions only (lower conviction)
- **SKIP**: Don't enter

### Risk Notes (orange/red warnings)
Shows specific risks:
- "Options bearish (PCR 0.49 bearish)" — options market positioned against you
- "MFI=24 — weak money flow" — no institutional accumulation
- "Dry volume (0.3x avg)" — thin market, unreliable signal

### Key Levels
| Level | Source | What it means |
|-------|--------|--------------|
| DAILY SMA | 20-day SMA | The "magnet" — mean reversion targets this |
| BB UPPER | Bollinger upper band | Near-term resistance proxy |
| BB LOWER | Bollinger lower band | Near-term support proxy |
| OI RESIST | Options OI wall | Heavy call writing = ceiling |
| OI SUPPORT | Options OI wall | Heavy put writing = floor |
| MAX PAIN | Options max pain | Price magnet near expiry |
| GEX HVL | Gamma exposure level | Dealer hedging pivot |

### Bottom Badges
PCR value, PIN risk level, GEX regime, DTE countdown.

---

## PANEL C: PRE-TRADE CHECKLIST (bottom right)

Each row is pass (✓ green), warn (! gold), or fail (✗ red):

| Check | Pass | Warn | Fail |
|-------|------|------|------|
| **WR(30)** | Deep (<-50): strong bounce | In zone (<-30) | >-30: not oversold |
| **vs SMA** | Deep pullback (<-3%) | Below SMA (<-1.5%) | Above SMA |
| **MFI** | Strong (≥50): accumulation | OK (≥30) | Weak (<30): no buyers |
| **Drawdown** | Capitulation (>10% off high) | Moderate (>5%) | Shallow (<5%) |
| **Red Streak** | Exhaustion (≥5 consecutive) | Selling (≥3 days) | No streak |
| **Options** | Overlay > +3 (supportive) | — | Overlay < -3 (bearish) |
| **Filing** | Bullish filing confirmed | — | TRAP detected |
| **Entry Verdict** | ▶ PRIMARY (FULL) | ▷ SECONDARY (HALF) | NO ENTRY (missing conditions) |

---

## SCORING BREAKDOWN

### Base Score (0-100)

| Factor | Max Points | Best Value |
|--------|-----------|------------|
| WR(30) depth | +25 | < -70 (deeply oversold) |
| Distance below SMA | +15 | < -5% |
| MFI value | +8 | ≥ 50 (strong accumulation) |
| MFI slope (rising) | +4 | Rising 3-bar slope |
| Drawdown from 50d high | +8 | ≤ -10% (capitulation) |
| Red streak | +6 | ≥ 5 consecutive red days |
| Volume ratio | +4 | ≥ 1.5x avg (institutional surge) |
| BBW percentile | +3 | < 30th (squeeze = coiled spring) |
| Above SMA penalty | -10 | If price above SMA |
| Weak MFI penalty | -5 | MFI < 30 |
| Dry volume penalty | -3 | Volume < 0.5x avg |
| Expanded BBW penalty | -2 | > 70th pctl |

### Options Overlay (±10)

| Condition | Points |
|-----------|--------|
| PCR ≤ 0.7 (bearish) | -5 |
| PCR ≤ 0.7 + GEX negative | -5 -3 = -8 |
| PCR ≥ 1.3 (bullish) | +3 |
| PCR ≥ 1.3 + GEX negative | +3 +2 = +5 |
| Near OI support (< 2%) | +2 |

### Filing Overlay (±10)

| Condition | Points |
|-----------|--------|
| Bullish filing, conviction ≥ 7 | +7 |
| Bullish filing, conviction ≥ 5 | +4 |
| Bearish filing + base ≥ 55 | -10 + TRAP (caps score at 30) |
| Bearish filing + base < 55 | -5 |

---

## ENTRY TIERS

### PRIMARY (FULL size)
All of these must be true simultaneously:
- WR(30) < -30 (oversold zone)
- Close < SMA(20) (in pullback)
- MFI > 30 (money flowing in)
- Drawdown > 5% from 50-day high
- ≥ 3 consecutive red candles (selling exhaustion)
- Volume > 0.5× 20-day average (not a ghost town)

**OOS stats:** Sharpe 4.17, Win 85.3%, Avg +2.44%, N=75

### SECONDARY (HALF size)
Core conditions only:
- WR(30) < -30
- Close < SMA(20)
- MFI > 30

Missing drawdown, red streak, or volume confirmation = lower conviction = half size.

---

## EXIT RULES

Three exits, checked in order each day:

1. **Profit Target (+3%)**: If price reaches +3% from entry → EXIT. Locks in gains before giveback. (Sweep showed trades average +2.8% peak then give back to +1.2% without PT.)

2. **BBW Contraction**: If held ≥5 days AND BB width is contracting (slope < 0) AND price is above SMA → EXIT. This means the bounce played out and volatility is fading.

3. **Max Hold (30 days)**: Hard cap. If neither PT nor BBW triggered in 30 trading days → EXIT regardless.

---

## SCANNER TAB COLUMNS

| Column | Meaning | Good values |
|--------|---------|-------------|
| Score | Final composite (base + overlays) | ≥75 STRONG, ≥60 GOOD |
| Tier | PRI (all conditions), SEC (core only), — (no entry) | PRI with ▶ prefix |
| Size | FULL / HALF / SKIP | FULL or HALF |
| WR | Williams %R (30-day) | < -50 emerald, < -30 green |
| MFI | Money Flow Index | ≥ 50 emerald, ≥ 30 green |
| vs SMA | % distance from 20-SMA | Red = below (good for MR) |
| DD% | Drawdown from 50-day high | Red < -10% (capitulation) |
| Red | Consecutive red candle days | ≥ 3 gold, ≥ 5 red |
| Regime | GEX-based options regime | TREND_FRIENDLY preferred |
| DTE | Days to options expiry | ≥ 5d safe |
| Reason | Entry reason or why not | Full signal breakdown |

---

## QUICK DECISION FRAMEWORK

### When to enter:
Scanner shows ▶ (filled triangle) with Score ≥ 70 → check dashboard → all checklist items green → enter at SIZE shown.

### When to wait:
Score 50-70, some checklist items yellow → conditions improving but not ripe yet.

### When to skip:
Score < 50, MFI weak, above SMA, or TRAP detected → no edge, wait for pullback.

### After entry:
Monitor for exit: did it hit +3%? Is BBW contracting while above SMA? Either → close position.

---

## EXAMPLE: LT (from your screenshot)

- **WR(30) = -96**: Deeply oversold ✓ (+25 points)
- **vs SMA = -12.0%**: Deep pullback ✓ (+15 points)
- **MFI = 24**: WEAK ✗ (-5 points) — nobody is buying this dip yet
- **DD = -22.6%**: Capitulation ✓ (+8 points)
- **Red streak = 0**: No exhaustion signal (! 0 points)
- **Options: PCR = 0.495 bearish, GEX positive**: -5 overlay

**Result:** base=71, opt=-5 → 66 GOOD SKIP

**Why SKIP despite GOOD score?** MFI < 30 kills the entry. The stock is deeply oversold (great for WR/SMA/DD) but there's no institutional accumulation yet. When MFI crosses above 30, the SECONDARY entry triggers. When MFI crosses 30 AND red streak hits 3+ days AND DD stays deep → PRIMARY triggers at FULL size.

**What to watch for:** MFI turning green (crossing 30). That's the signal that institutions have started buying the dip.
