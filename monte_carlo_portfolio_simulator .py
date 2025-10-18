#!/xsr/bin/env python3
"""
Monte Carlo Portfolio Simulator (BTC, VOO, VUG, GLDM) vs S&P 500

What this script does
- Pulls historical daily prices from Polygon.io for BTC (X:BTCUSD), VOO, VUG, GLDM, and S&P 500 (I:SPX with SPY fallback).
- Computes daily **log returns** and estimates a **multivariate Normal** (mean + covariance) from history.
- Runs Monte Carlo simulations to project future portfolio paths preserving cross-asset correlations.
- Simulates S&P 500 separately from its own return distribution for a side‑by‑side comparison.
- Outputs: a **line chart** comparing the projected median of your portfolio vs. S&P 500 and a printed stats block (Sharpe, CAGR, vol, VaR, CVaR, drawdowns, etc.).

"""
import argparse
import os
import sys
import time
import math
from datetime import datetime, date
from typing import Dict, List, Tuple
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # saves PNG without GUI
import matplotlib.pyplot as plt


# Config & Helpers

DEFAULT_API_KEY = os.getenv("POLYGON_API_KEY", "__xM31vthosrQau8asNbRQdjYjzcjnJh")
ASSET_MAP = {
    "BTC": "X:BTCUSD",   # Polygon crypto ticker format
    "VOO": "VOO",
    "VUG": "VUG",
    "GLDM": "GLDM",
}
SP500_TICKERS = ["I:SPX", "SPY"]  # try index first, then ETF fallback

SESSION = requests.Session()
BASE = "https://api.polygon.io"


def _fmt_exc(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"


def fmt_pct(x: float) -> str:
    """Format a decimal as a percentage with two decimals."""
    return f"{x*100:6.2f}%"


def daterange_str(d: str) -> str:
    # Validate YYYY-MM-DD
    try:
        datetime.strptime(d, "%Y-%m-%d")
        return d
    except ValueError:
        raise SystemExit(f"Invalid date '{d}'. Use YYYY-MM-DD.")



# Polygon data fetch


def fetch_polygon_aggregates(
    ticker: str,
    start: str,
    end: str,
    api_key: str,
    multiplier: int = 1,
    timespan: str = "day",
    adjusted: bool = True,
    limit: int = 50000,
    max_pages: int = 100,
) -> pd.DataFrame:
    """Fetch adjusted daily bars for a ticker from Polygon v2 aggregates API.
    Returns a DataFrame with columns: [date, close].
    Follows pagination using `next_url` if present.
    """
    url = f"{BASE}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start}/{end}"
    params = {
        "adjusted": str(adjusted).lower(),
        "sort": "asc",
        "limit": limit,
        "apiKey": api_key,
    }

    all_rows = []
    pages = 0
    next_url = url
    next_params = params.copy()

    while next_url and pages < max_pages:
        resp = SESSION.get(next_url, params=next_params, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Polygon error {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        results = data.get("results", [])
        for r in results:
            # 't' in ms epoch; 'c' is close price
            ts = pd.to_datetime(r["t"], unit="ms", utc=True).tz_convert(None).normalize()
            close = float(r["c"]) if r.get("c") is not None else np.nan
            all_rows.append((ts.date(), close))

        next_url = data.get("next_url")
        # After first page, polygon requires apiKey on the next_url request via param again
        next_params = {"apiKey": api_key} if next_url else {}
        pages += 1
        if pages > 1:
            time.sleep(0.1)  

    if not all_rows:
        raise RuntimeError(f"No results for {ticker} in {start}..{end}.")

    df = pd.DataFrame(all_rows, columns=["date", "close"]).dropna()
    df = df.groupby("date", as_index=False).last().sort_values("date")
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df


def get_price_series(ticker: str, start: str, end: str, api_key: str, adjusted: bool = True) -> pd.Series:
    df = fetch_polygon_aggregates(ticker, start, end, api_key, adjusted=adjusted)
    return df["close"].astype(float)


def get_sp500_series(start: str, end: str, api_key: str) -> Tuple[str, pd.Series]:
    last_err = None
    for t in SP500_TICKERS:
        try:
            s = get_price_series(t, start, end, api_key, adjusted=True)
            return t, s
        except Exception as e:
            last_err = e
            continue
    raise SystemExit(f"Failed to fetch S&P 500 via {SP500_TICKERS}. Last error: {_fmt_exc(last_err)}")



# Math & Metrics


def to_log_returns(price: pd.Series) -> pd.Series:
    return np.log(price / price.shift(1)).dropna()


def align_returns(returns_map: Dict[str, pd.Series]) -> pd.DataFrame:
    df = pd.DataFrame(returns_map)
    # Drop dates with any NaN to make covariance well-defined
    return df.dropna().copy()


def portfolio_stats(daily_ret: pd.Series, rf_annual: float = 0.0) -> Dict[str, float]:
    # daily_ret = portfolio log returns
    mu_d = daily_ret.mean()
    sd_d = daily_ret.std(ddof=1)
    rf_d = rf_annual / 252.0

    sharpe = (mu_d - rf_d) / (sd_d + 1e-12) * math.sqrt(252)

    # Build equity curve
    eq = np.exp(daily_ret.cumsum())

    # CAGR
    years = len(daily_ret) / 252.0
    cagr = float(eq.iloc[-1]) ** (1 / max(years, 1e-9)) - 1.0

    # Vol annualized
    vol_ann = sd_d * math.sqrt(252)

    # Max drawdown
    roll_max = eq.cummax()
    dd = (eq / roll_max) - 1.0
    max_dd = dd.min()

    # Sortino (downside deviation against 0 per day)
    downside = daily_ret[daily_ret < 0]
    dd_std = downside.std(ddof=1)
    sortino = (mu_d - rf_d) / (dd_std + 1e-12) * math.sqrt(252)

    return {
        "Daily mean (log)": mu_d,
        "Daily stdev": sd_d,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Vol (ann)": vol_ann,
        "CAGR": cagr,
        "Max Drawdown": max_dd,
        "Length (days)": len(daily_ret),
    }


def var_cvar_from_paths(terminal_values: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    """Compute VaR/CVaR on terminal returns given terminal *values* (start=1)."""
    if terminal_values.ndim == 1:
        end_ret = terminal_values - 1.0
    else:
        end_ret = terminal_values - 1.0
    q = np.quantile(end_ret, alpha)
    cvar = end_ret[end_ret <= q].mean() if np.any(end_ret <= q) else q
    return float(q), float(cvar)



# Monte Carlo


def simulate_paths(
    mean_vec: np.ndarray,
    cov: np.ndarray,
    weights: np.ndarray,
    days: int,
    sims: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate correlated asset & portfolio paths.
    Returns (portfolio_paths, asset_paths) with starting value 1.0.
    """
    rng = np.random.default_rng(seed)
    n = len(mean_vec)
    draws = rng.multivariate_normal(mean_vec, cov, size=(sims, days))  # (sims, days, n)
    # Asset cumulative log-return paths
    cumlog = draws.cumsum(axis=1)
    asset_paths = np.exp(np.concatenate([np.zeros((sims, 1, n)), cumlog], axis=1))
    # Portfolio cumulative log-return (weighted sum of logs)
    port_cumlog = (draws @ weights).cumsum(axis=1)
    port_paths = np.exp(np.concatenate([np.zeros((sims, 1)), port_cumlog], axis=1))
    return port_paths, asset_paths


def simulate_univariate(mean: float, var: float, days: int, sims: int, seed: int = 43) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = rng.normal(mean, math.sqrt(var), size=(sims, days))
    cumlog = draws.cumsum(axis=1)
    paths = np.exp(np.concatenate([np.zeros((sims, 1)), cumlog], axis=1))
    return paths



# Main script


def parse_weights(w_str: str, assets: List[str]) -> Dict[str, float]:
    if not w_str or w_str.lower() == "equal":
        return {a: 1.0 / len(assets) for a in assets}
    parts = [p.strip() for p in w_str.split(",") if p.strip()]
    out: Dict[str, float] = {}
    for p in parts:
        if ":" not in p:
            raise SystemExit("Weights must be like BTC:0.2,VOO:0.4,VUG:0.3,GLDM:0.1")
        k, v = p.split(":", 1)
        k = k.strip().upper()
        v = float(v.strip())
        if k not in assets:
            raise SystemExit(f"Unknown asset in weights: {k}")
        out[k] = v
    s = sum(out.get(a, 0.0) for a in assets)
    if s <= 0:
        raise SystemExit("Weights must sum to > 0")
    return {a: out.get(a, 0.0) / s for a in assets}


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo portfolio simulator vs S&P 500 (Polygon.io)")
    parser.add_argument("--start", default="2015-01-01", type=daterange_str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=date.today().isoformat(), type=daterange_str, help="End date YYYY-MM-DD")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="Polygon API Key (or env POLYGON_API_KEY)")
    parser.add_argument("--sims", type=int, default=5000, help="Number of Monte Carlo simulations")
    parser.add_argument("--horizon", type=int, default=252, help="Future horizon in trading days")
    parser.add_argument("--weights", default="equal", help="Weights like BTC:0.2,VOO:0.4,VUG:0.3,GLDM:0.1 or 'equal'")
    parser.add_argument("--rf", type=float, default=0.0, help="Annual risk-free rate (e.g., 0.02 for 2%)")
    parser.add_argument("--initial", type=float, default=10000.0, help="Starting portfolio value for charts")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed for reproducibility")
    parser.add_argument("--save-csv", default=None, help="Optional path to save fetched price history CSV")
    parser.add_argument("--out", default="monte_carlo_vs_sp500.png", help="Output chart filename")
    args = parser.parse_args()

    assets = ["BTC", "VOO", "VUG", "GLDM"]
    weights_map = parse_weights(args.weights, assets)
    w = np.array([weights_map[a] for a in assets])

    # Fetch historical closes
    print("Fetching historical prices from Polygon...")
    closes: Dict[str, pd.Series] = {}
    for a in assets:
        t = ASSET_MAP[a]
        print(f"  - {a} ({t}) ...", end="", flush=True)
        s = get_price_series(t, args.start, args.end, args.api_key, adjusted=True)
        print(f" {len(s)} bars")
        closes[a] = s

    spx_ticker, spx_close = get_sp500_series(args.start, args.end, args.api_key)
    print(f"  - S&P 500 benchmark via {spx_ticker}: {len(spx_close)} bars")

    # Optionally save history
    if args.save_csv:
        combo = pd.DataFrame({**closes, "SP500": spx_close}).dropna()
        combo.to_csv(args.save_csv, index_label="date")
        print(f"Saved history to {args.save_csv}")

    # Compute daily log returns and align
    rets_map = {k: to_log_returns(v) for k, v in closes.items()}
    port_df = align_returns(rets_map)

    spx_ret = to_log_returns(spx_close)
    spx_ret = spx_ret.loc[port_df.index.intersection(spx_ret.index)]
    port_df = port_df.loc[spx_ret.index]

    if len(port_df) < 30:
        raise SystemExit("Not enough overlapping data across assets and SPX to estimate distribution.")

    # Estimate distribution
    mu = port_df.mean().values  # per-asset daily log-return mean
    cov = port_df.cov().values  # daily covariance matrix

    # Historical portfolio daily log return (for historical stats)
    hist_port_logret = (port_df @ w)

    # Historical stats
    hist_stats = portfolio_stats(hist_port_logret, rf_annual=args.rf)
    spx_hist_stats = portfolio_stats(spx_ret.loc[hist_port_logret.index], rf_annual=args.rf)

    # Correlation matrix (historical)
    corr = port_df.corr()

    # Monte Carlo simulate future paths starting at value=1
    print(f"Simulating {args.sims} paths over {args.horizon} days...")
    port_paths, _ = simulate_paths(mu, cov, w, days=args.horizon, sims=args.sims, seed=args.seed)

    # Simulate SPX univariately
    spx_mu = float(spx_ret.mean())
    spx_var = float(spx_ret.var(ddof=1))
    spx_paths = simulate_univariate(spx_mu, spx_var, days=args.horizon, sims=args.sims, seed=args.seed + 1)

    # Terminal distribution & risk measures
    port_terminal = port_paths[:, -1]
    spx_terminal = spx_paths[:, -1]

    port_p50 = np.quantile(port_paths, 0.50, axis=0)
    spx_p50 = np.quantile(spx_paths, 0.50, axis=0)

    # VaR & CVaR at horizon
    var5, cvar5 = var_cvar_from_paths(port_terminal, alpha=0.05)
    var5_spx, cvar5_spx = var_cvar_from_paths(spx_terminal, alpha=0.05)

    # Prob of loss
    p_loss = float((port_terminal < 1.0).mean())
    p_loss_spx = float((spx_terminal < 1.0).mean())


    # Chart: line comparison (median paths)
  
    initial = float(args.initial)
    x = np.arange(port_p50.shape[0])

    plt.figure(figsize=(11, 6.5))
    plt.title("Projected Median Value: Portfolio vs S&P 500")
    plt.plot(x, initial * port_p50, label="Portfolio (median)")
    plt.plot(x, initial * spx_p50, linestyle="--", label="S&P 500 (median)")
    plt.xlabel("Days ahead")
    plt.ylabel(f"Value (start = {initial:,.0f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    plt.savefig(out_path, dpi=140)
    print(f"Saved chart to {out_path}")


    # Console stats

    print("==== Historical (based on overlap window) ====")
    print(f"Length (days): {hist_stats['Length (days)']}")
    print(
        f"CAGR:   {fmt_pct(hist_stats['CAGR'])} | Vol: {fmt_pct(hist_stats['Vol (ann)'])} | "
        f"Sharpe: {hist_stats['Sharpe']:.2f} | Sortino: {hist_stats['Sortino']:.2f}"
    )
    print(f"Max DD: {fmt_pct(hist_stats['Max Drawdown'])}")

    print("S&P 500 (historical on same dates):")
    print(
        f"CAGR:   {fmt_pct(spx_hist_stats['CAGR'])} | Vol: {fmt_pct(spx_hist_stats['Vol (ann)'])} | "
        f"Sharpe: {spx_hist_stats['Sharpe']:.2f} | Sortino: {spx_hist_stats['Sortino']:.2f}"
    )
    print(f"Max DD: {fmt_pct(spx_hist_stats['Max Drawdown'])}")

    # Terminal distribution stats
    port_median = float(np.median(port_terminal) - 1.0)
    spx_median = float(np.median(spx_terminal) - 1.0)

    print("==== Simulated horizon (terminal, start=1) ====")
    print(
        f"Portfolio median: {fmt_pct(port_median)} | "
        f"5th pct: {fmt_pct(float(np.quantile(port_terminal-1, 0.05)))} | "
        f"95th pct: {fmt_pct(float(np.quantile(port_terminal-1, 0.95)))}"
    )
    print(
        f"Portfolio VaR(5%): {fmt_pct(var5)} | CVaR(5%): {fmt_pct(cvar5)} | P(Loss): {p_loss*100:.2f}%"
    )

    print(
        f"S&P 500 median:  {fmt_pct(spx_median)} | "
        f"5th pct: {fmt_pct(float(np.quantile(spx_terminal-1, 0.05)))} | "
        f"95th pct: {fmt_pct(float(np.quantile(spx_terminal-1, 0.95)))}"
    )
    print(
        f"S&P 500 VaR(5%): {fmt_pct(var5_spx)} | CVaR(5%): {fmt_pct(cvar5_spx)} | P(Loss): {p_loss_spx*100:.2f}%"
    )

    # Weights and correlation
    print("Weights (normalized):")
    for a in assets:
        print(f"  {a}: {weights_map[a]:.4f}")

    print("Asset correlation (historical):")
    print(corr.to_string(float_format=lambda x: f"{x: .3f}"))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Aborted by user.")
    except Exception as e:
        print(f"Error: {_fmt_exc(e)}", file=sys.stderr)
        sys.exit(1)
