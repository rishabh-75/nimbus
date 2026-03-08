# NSE Trading Signal Dashboard 📈

A production-ready daily trading signal dashboard for NSE options markets.
Monitors options wall structures and institutional block/bulk deal activity to
generate actionable intraday and swing trading signals.

---

## Features

| Feature | Description |
|---|---|
| **Options Wall Analysis** | Cross-expiry OI aggregation, call/put walls, PCR, IV smile, max pain |
| **Institutional Activity** | FII/DII/Promoter deal detection, proximity scoring, zone classification |
| **Signal Engine** | Composite bullish/bearish score from options + institutional + IV axes |
| **Alerts** | Real-time toasts for score spikes, OI moves, PCR crossovers, signal flips |
| **Auto-Refresh** | APScheduler every 5 min during market hours (09:15–15:30 IST) |
| **Demo Mode** | Synthetic SBIN data generated automatically when no CSVs are present |
| **Dark Theme** | Full dark UI with #0d0d1a background and Plotly interactive charts |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the dashboard

```bash
streamlit run app.py --server.port 8501 --theme.base dark
```

### 3. Open in browser

```
http://localhost:8501
```

The app loads **demo data automatically** if no CSVs are present.

---

## Data Format

### Options Chain CSV

Place files in `./data/options/` with naming pattern:

```
{SYMBOL}_options_{YYYYMMDD}.csv
```

Example: `SBIN_options_20241210.csv`

Required columns:

| Column | Description |
|---|---|
| `Strike` | Strike price (numeric) |
| `Expiry` | Expiry date (string, e.g. "28-Nov-2024") |
| `OptionType` | "CE" or "PE" |
| `OpenInterest` | Open interest (numeric) |
| `Volume` | Volume traded |
| `IV` | Implied volatility (%) |
| `LTP` | Last traded price |

### Bulk / Block Deals CSV

Place files in `./data/deals/` **or** download directly from NSE via the sidebar button.

Required columns:

| Column | Description |
|---|---|
| `Date` | Trade date |
| `Symbol` | NSE symbol |
| `Client Name` | Entity name |
| `Buy/Sell` | "BUY" or "SELL" |
| `Quantity Traded` | Shares traded |
| `Trade Price/Wght. Avg. Price` | Price per share |

NSE URLs (auto-downloaded):
- Bulk: `https://archives.nseindia.com/content/equities/bulk.csv`
- Block: `https://archives.nseindia.com/content/equities/block.csv`

### Price / OHLCV CSV

Place in `./data/price/` as `{SYMBOL}_price.csv`:

```
Date,Open,High,Low,Close,Volume
2024-12-10,592,598,588,595,1234567
```

---

## Dashboard Tabs

### 🏦 Tab 1 — Wall Overview
- KPI cards: PCR, Sentiment, Max Pain, Support, Resistance
- Interactive OI wall chart (calls up / puts down)
- PCR by strike, IV smile, total OI area, wall strength bars

### 🕵️ Tab 2 — Insider Activity
- KPI cards: deals count, at-wall count, high-conviction, top entity
- OI wall + deal overlay (triangles sized by quantity)
- Signal score scatter, net buy/sell per level
- Full deals table with colour coding + CSV export

### 🎯 Tab 3 — Signal Dashboard
- **Master Signal Card**: 🟢 BULLISH / 🔴 BEARISH / 🟡 NEUTRAL with confidence %
- Three panels: Options Signal, Institutional Signal, Composite Gauge
- Zone table with all accumulation/distribution zones
- Alert log with timestamp

### 📊 Tab 4 — Historical Comparison
- OI change chart (today vs yesterday)
- PCR trend line across sessions
- Max Pain migration chart
- New OI buildup table (strikes with >10% OI increase)

### ⚙️ Tab 5 — Settings & Data Health
- Data quality report (null counts, coverage %)
- Raw data preview (options / deals / walls)
- Session info, refresh schedule

---

## Signal Logic

### Options Signal (–100 to +100)

| Condition | Score |
|---|---|
| PCR < 0.7 | +30 |
| PCR 0.7–1.0 | +15 |
| PCR 1.0–1.3 | –15 |
| PCR > 1.3 | –30 |
| CMP < Max Pain | +20 |
| CMP > Max Pain | –20 |
| Put IV > Call IV by >3 | +15 |
| Call IV > Put IV by >3 | –15 |
| Put Wall below CMP | +15 |
| Call Wall above CMP | –15 |

### Institutional Signal (–100 to +100)

| Condition | Score |
|---|---|
| Each Accumulation zone (score ≥60) | +15 (max +45) |
| Each Distribution zone (score ≥60) | –15 (max –45) |
| Promoter BUY | +20 |
| Promoter SELL | –20 |
| Net FII+DII positive | +10 |
| Score ≥80 institutional BUY | +15 |

### Composite Signal

```
Final = Options × 40% + Institutional × 40% + IV Skew × 20%
```

---

## Deal Scoring (0–100)

| Axis | Max Points |
|---|---|
| Proximity to wall (closer = higher) | 30 |
| Wall OI size | 20 |
| Deal size (₹ value) | 20 |
| Entity quality (Promoter > FII > DII > Broker > Retail) | 20 |
| Directional alignment (BUY at Put Wall / SELL at Call Wall) | 10 |

---

## Alert Types

| Alert | Trigger |
|---|---|
| 🟢/🔴 Score Alert | Any deal scores ≥ threshold (configurable, default 70) |
| ⚠️ OI Spike | OI at any strike changes >20% vs last load |
| 📊 PCR Crossover | PCR crosses 0.7 or 1.3 |
| 🎯 Max Pain Migration | Max pain moves by ≥1 strike |
| 🔄 Signal Flip | Composite signal changes direction |

---

## Sidebar Controls

| Control | Default | Description |
|---|---|---|
| Symbol | SBIN | NSE trading symbol |
| Date | Today | Data date |
| Wall OI Percentile | 75 | Wall identification threshold |
| Proximity Tolerance | 1.5% | Deal-to-wall matching radius |
| Min Signal Score | 40 | Hide deals below this score |
| Score Alert | 70 | Toast alert threshold |
| OI Spike | 20% | OI change alert threshold |

---

## Project Structure

```
nse_dashboard/
├── app.py                    ← Streamlit entry point
├── requirements.txt
├── README.md
├── modules/
│   ├── __init__.py
│   ├── options_wall.py       ← OptionsWallCalculator
│   ├── insider_detector.py   ← InsiderWallDetector + scoring
│   ├── signal_engine.py      ← SignalEngine
│   ├── data_manager.py       ← File watching + NSE download + demo data
│   ├── alert_manager.py      ← AlertManager
│   └── chart_builder.py      ← All Plotly chart functions
└── data/
    ├── options/              ← Drop options CSVs here
    ├── deals/                ← Drop bulk/block deal CSVs here
    └── price/                ← Drop OHLCV CSVs here
```

---

## Notes

- The app runs fully in **demo mode** with synthetic SBIN data when no CSVs are present.
- All charts are interactive Plotly figures — hover, zoom, pan supported.
- NSE deal download requires internet access and may be rate-limited by NSE servers.
- For real-time data, connect to a broker API and write CSVs to the data folders.
- For desktop notifications, install `plyer`: `pip install plyer`
