# Monte Carlo Portfolio Simulator (Polygon.io)

A Python tool that simulates future performance of a BTC / VOO / VUG / GLDM portfolio using Monte Carlo. It estimates historical **log-return** mean/covariance from Polygon.io data, preserves cross-asset correlations, and compares the portfolio’s projected path to the S&P 500.

---

## Features & Approach
- **Polygon.io integration** for BTC, VOO, VUG, GLDM, and S&P 500 (I:SPX with SPY fallback)  
- **Multivariate Normal** fit on daily log returns (mean + covariance)  
- **Correlated Monte Carlo** paths for the portfolio; separate S&P simulation  
- **Metrics:** Sharpe, Sortino, CAGR, annualized vol, max drawdown, VaR/CVaR, loss probability, correlation matrix  
- **Visualization:** line chart of projected median **Portfolio vs S&P 500** (PNG saved headlessly)

---

## Getting Started

### Prerequisites
- Python 3.9+  
- Packages: `requests`, `numpy`, `pandas`, `matplotlib`

```bash
pip install -r requirements.txt
# or
pip install requests numpy pandas matplotlib
```

### Polygon API Key
```bash
export POLYGON_API_KEY="your_polygon_api_key"
```

---

## Usage

```bash
python monte_carlo_portfolio_simulator.py   --start 2018-01-01 --end 2025-09-13   --sims 10000 --horizon 252   --weights BTC:0.20,VOO:0.40,VUG:0.30,GLDM:0.10   --rf 0.02 --initial 10000   --out monte_carlo_vs_sp500.png
```

**Common flags**
- `--start/--end` : historical window for estimating returns  
- `--sims` : number of Monte Carlo paths (default 5000)  
- `--horizon` : days to project (default 252)  
- `--weights` : e.g., `BTC:0.2,VOO:0.4,VUG:0.3,GLDM:0.1` or `equal`  
- `--rf` : annual risk-free rate (e.g., `0.02`)  
- `--initial` : starting portfolio value for charts  
- `--out` : output PNG filename (saved to current working directory)

---

## Output
- **Chart:** `monte_carlo_vs_sp500.png` – median **Portfolio vs S&P 500** projected values  
- **Console stats:** CAGR, vol, Sharpe, Sortino, max drawdown, VaR/CVaR, loss probability, and the asset correlation matrix

---

## Methodology & Notes
- Daily **log returns** computed from adjusted closes; assets aligned to overlapping dates  
- Portfolio paths use correlated draws from the fitted multivariate Normal  
- S&P 500 simulated univariately from its historical log-return mean/variance  
- Assumes no fees, slippage, taxes, or rebalancing during the projection window

---

## Limitations
- Normality can understate fat tails (especially for BTC)  
- Results depend on historical window choice and data quality  
- Not investment advice

---

## License
MIT
