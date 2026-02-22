"""
api.py — Portfolio Optimization REST API
Connects the HTML frontend to real yfinance data + MPT calculations.

Install deps:  pip install flask flask-cors yfinance numpy scipy
Run locally:   python api.py
Deploy:        Add this as a second Render Web Service (separate from Streamlit)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import numpy as np
from scipy.optimize import minimize
import traceback

app = Flask(__name__)
CORS(app)   # Allow requests from your HTML frontend domain


# ─────────────────────────────────────────────
# HELPER: Download price data & compute μ, Σ
# ─────────────────────────────────────────────
def get_market_data(tickers: list, period_days: int, frequency: str) -> dict:
    """
    Download historical prices from yfinance and compute:
      - Annual expected returns (μ) via geometric mean
      - Annual covariance matrix (Σ) scaled by trading periods
    """
    # Map frequency string to yfinance interval and annualisation factor
    freq_map = {
        "252": ("1d",  252),   # daily
        "52":  ("1wk", 52),    # weekly
        "12":  ("1mo", 12),    # monthly
    }
    interval, periods_per_year = freq_map.get(str(frequency), ("1d", 252))

    # Convert days to yfinance period string
    if period_days <= 90:
        yf_period = "3mo"
    elif period_days <= 180:
        yf_period = "6mo"
    elif period_days <= 365:
        yf_period = "1y"
    elif period_days <= 1825:
        yf_period = "5y"
    else:
        yf_period = "10y"

    # Download adjusted close prices
    raw = yf.download(
        tickers,
        period=yf_period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    # Handle single vs multiple ticker response
    if len(tickers) == 1:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    else:
        prices = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.xs("Close", axis=1, level=1)

    prices = prices.dropna(how="all")

    # Compute log returns for each ticker
    log_returns = np.log(prices / prices.shift(1)).dropna()

    # Annual expected return: geometric mean = exp(mean(log_ret) * periods_per_year) - 1
    mu = np.exp(log_returns.mean() * periods_per_year) - 1

    # Annual covariance matrix
    cov = log_returns.cov() * periods_per_year

    # Return valid tickers only (some may fail to download)
    valid = [t for t in tickers if t in mu.index and not np.isnan(mu[t])]

    return {
        "mu":     mu[valid].values.tolist(),
        "cov":    cov.loc[valid, valid].values.tolist(),
        "tickers": valid,
    }


# ─────────────────────────────────────────────
# HELPER: MPT Optimization (scipy SLSQP)
# ─────────────────────────────────────────────
def optimise_portfolio(mu: np.ndarray, cov: np.ndarray,
                       rfr: float, allow_short: bool,
                       n_frontier: int = 60) -> dict:
    n = len(mu)

    # ── Portfolio metric functions (exact formulas)
    port_return   = lambda w: float(np.dot(w, mu))
    port_variance = lambda w: float(w @ cov @ w)
    port_vol      = lambda w: float(np.sqrt(max(0.0, port_variance(w))))
    port_sharpe   = lambda w: (port_return(w) - rfr) / port_vol(w) if port_vol(w) > 1e-9 else -1e9

    # ── Constraints & bounds
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds      = ((-1.0, 1.0) if allow_short else (0.0, 1.0),) * n
    eq_weight   = np.ones(n) / n

    def run_optimise(objective, init_w):
        result = minimize(
            objective,
            init_w,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 2000},
        )
        return result.x if result.success else init_w

    # ── Max Sharpe (tangency portfolio)
    neg_sharpe  = lambda w: -port_sharpe(w)
    best_w, best_s = eq_weight.copy(), -1e9
    starts = [eq_weight] + [
        np.random.dirichlet(np.ones(n)) for _ in range(8)
    ]
    for s0 in starts:
        w = run_optimise(neg_sharpe, s0)
        s = port_sharpe(w)
        if s > best_s:
            best_s, best_w = s, w.copy()

    opt_w   = best_w
    opt_ret = port_return(opt_w)
    opt_vol = port_vol(opt_w)
    opt_sharpe = best_s

    # ── Equal-weight baseline
    eq_ret    = port_return(eq_weight)
    eq_vol    = port_vol(eq_weight)
    eq_sharpe = port_sharpe(eq_weight)

    # ── Efficient frontier: sweep target returns, minimise variance
    mu_min = float(np.min(mu))
    mu_max = float(np.max(mu))
    frontier_pts = []

    for i in range(n_frontier + 1):
        target_mu = mu_min + i * (mu_max - mu_min) / n_frontier
        constraints_mv = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target_mu: port_return(w) - t},
        ]
        w_mv = run_optimise(port_variance, eq_weight.copy())
        # Re-run with return constraint
        res = minimize(
            port_variance, eq_weight,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints_mv,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if res.success:
            frontier_pts.append({
                "vol": float(port_vol(res.x)),
                "ret": float(port_return(res.x)),
            })

    # Sort by vol, keep only upper (efficient) portion
    frontier_pts.sort(key=lambda p: p["vol"])
    efficient = []
    max_ret   = -1e9
    for p in frontier_pts:
        if p["ret"] >= max_ret - 0.001:
            efficient.append(p)
            max_ret = max(max_ret, p["ret"])

    return {
        "optWeights": opt_w.tolist(),
        "optReturn":  opt_ret,
        "optVol":     opt_vol,
        "optSharpe":  opt_sharpe,
        "eqReturn":   eq_ret,
        "eqVol":      eq_vol,
        "eqSharpe":   eq_sharpe,
        "frontier":   efficient,
        "mu":         mu.tolist(),
        "sigma":      [float(np.sqrt(cov[i][i])) for i in range(n)],
    }


# ─────────────────────────────────────────────
# ROUTE: POST /optimise
# ─────────────────────────────────────────────
@app.route("/optimise", methods=["POST"])
def optimise():
    """
    Request body (JSON):
    {
      "tickers":   ["AAPL", "MSFT", "GOOG"],
      "rfr":       0.02,
      "frequency": "252",
      "period":    365,
      "allowShort": false
    }

    Response (JSON):
    {
      "tickers":    ["AAPL", "MSFT", "GOOG"],
      "optWeights": [0.45, 0.35, 0.20],
      "optReturn":  0.182,
      "optVol":     0.163,
      "optSharpe":  0.994,
      "eqReturn":   0.140,
      "eqVol":      0.198,
      "eqSharpe":   0.606,
      "frontier":   [{"vol": 0.14, "ret": 0.09}, ...],
      "mu":         [0.20, 0.18, 0.15],
      "sigma":      [0.28, 0.22, 0.25]
    }
    """
    try:
        body       = request.get_json(force=True)
        tickers    = [t.upper().strip() for t in body.get("tickers", [])]
        rfr        = float(body.get("rfr", 0.02))
        frequency  = str(body.get("frequency", "252"))
        period     = int(body.get("period", 365))
        allow_short = bool(body.get("allowShort", False))

        if len(tickers) < 2:
            return jsonify({"error": "Need at least 2 tickers"}), 400
        if len(tickers) > 20:
            return jsonify({"error": "Max 20 tickers"}), 400

        # Fetch real market data
        market = get_market_data(tickers, period, frequency)
        if len(market["tickers"]) < 2:
            return jsonify({"error": "Could not fetch data for enough tickers. Check ticker symbols."}), 400

        mu  = np.array(market["mu"])
        cov = np.array(market["cov"])
        valid_tickers = market["tickers"]

        # Run MPT optimization
        result = optimise_portfolio(mu, cov, rfr, allow_short)
        result["tickers"] = valid_tickers

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
