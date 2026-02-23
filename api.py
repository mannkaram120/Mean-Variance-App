"""
api.py — Flask REST API for the portfolio optimizer HTML frontend.
Has its own yfinance download function (bypasses st.cache_data decorator in app.py).
"""

import os
import hashlib
import json
import time
import traceback
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, request
from flask_cors import CORS

# Import only the pure-math functions from app.py (no Streamlit decorators)
from app import (
    build_efficient_frontier,
    compute_annual_metrics,
    max_sharpe_weights,
    portfolio_return,
    portfolio_volatility,
)

app = Flask(__name__)
CORS(app)

# ── In-memory cache: same request returns instantly within 1 hour
_cache = {}
CACHE_TTL = 3600


def make_key(tickers, rfr, frequency, period_days, allow_short):
    raw = json.dumps([sorted(tickers), rfr, frequency, period_days, allow_short])
    return hashlib.md5(raw.encode()).hexdigest()


def get_cached(key):
    if key in _cache:
        ts, data = _cache[key]
        if time.time() - ts < CACHE_TTL:
            return data
        del _cache[key]
    return None


def set_cached(key, data):
    if len(_cache) >= 50:
        oldest = min(_cache, key=lambda k: _cache[k][0])
        del _cache[oldest]
    _cache[key] = (time.time(), data)


def download_prices(tickers: list, start_date: date, end_date: date) -> pd.DataFrame:
    """
    Download adjusted close prices using yfinance.
    Uses auto_adjust=True which is the correct approach for newer yfinance versions.
    Returns a DataFrame with tickers as columns and dates as index.
    """
    data = yf.download(
        tickers=tickers,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        auto_adjust=True,      # gives adjusted prices directly in "Close"
        progress=False,
        threads=True,
    )

    if data.empty:
        raise ValueError("No data returned from yfinance. Check ticker symbols.")

    # With auto_adjust=True, prices are in "Close" column
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    else:
        # Single ticker returns flat DataFrame
        prices = data[["Close"]].copy()
        prices.columns = [tickers[0]]

    # Drop columns/rows that are all NaN
    prices = prices.dropna(axis=1, how="all").dropna(how="all")

    if prices.empty:
        raise ValueError(
            "No valid price data found. Tickers may be invalid or "
            "have no data in the selected date range."
        )

    if prices.shape[1] < 2:
        raise ValueError(
            f"Only got data for {prices.shape[1]} ticker(s). "
            "Need at least 2 valid tickers to optimize."
        )

    return prices


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/optimise", methods=["POST"])
def optimise():
    try:
        body        = request.get_json(force=True)
        tickers     = [t.strip().upper() for t in body.get("tickers", [])]
        rfr         = float(body.get("rfr", 0.02))
        frequency   = str(body.get("frequency", "252"))
        period_days = int(body.get("period", 365))
        allow_short = bool(body.get("allowShort", False))

        if len(tickers) < 2:
            return jsonify({"error": "Need at least 2 tickers"}), 400
        if len(tickers) > 20:
            return jsonify({"error": "Maximum 20 tickers supported"}), 400

        # Return cached result instantly if available
        ck     = make_key(tickers, rfr, frequency, period_days, allow_short)
        cached = get_cached(ck)
        if cached:
            cached["cached"] = True
            return jsonify(cached)

        freq_map         = {"252": 252, "52": 52, "12": 12}
        periods_per_year = freq_map.get(frequency, 252)
        end_date         = date.today()
        start_date       = end_date - timedelta(days=period_days)

        # Download prices using our own function (not app.py's st.cache_data version)
        prices_df     = download_prices(tickers, start_date, end_date)
        valid_tickers = list(prices_df.columns)
        n             = len(valid_tickers)

        # Annualised μ and Σ
        mean_returns, cov_matrix = compute_annual_metrics(prices_df, periods_per_year)
        bounds = tuple((-1.0, 1.0) if allow_short else (0.0, 1.0) for _ in range(n))

        # Max Sharpe via SLSQP
        opt_w      = max_sharpe_weights(mean_returns, cov_matrix, rfr, bounds)
        opt_ret    = portfolio_return(opt_w, mean_returns)
        opt_vol    = portfolio_volatility(opt_w, cov_matrix)
        opt_sharpe = (opt_ret - rfr) / opt_vol if opt_vol > 0 else 0.0

        # Equal-weight baseline
        eq_w      = np.ones(n) / n
        eq_ret    = portfolio_return(eq_w, mean_returns)
        eq_vol    = portfolio_volatility(eq_w, cov_matrix)
        eq_sharpe = (eq_ret - rfr) / eq_vol if eq_vol > 0 else 0.0

        # Efficient frontier — 25 points (fast enough, smooth enough)
        target_returns = np.linspace(float(mean_returns.min()), float(mean_returns.max()), num=25)
        frontier_df    = build_efficient_frontier(mean_returns, cov_matrix, target_returns, rfr, bounds)

        frontier_pts = sorted(
            [{"vol": float(r["target_volatility"]), "ret": float(r["target_return"])}
             for _, r in frontier_df.iterrows()],
            key=lambda p: p["vol"]
        )
        efficient, max_r = [], -1e9
        for p in frontier_pts:
            if p["ret"] >= max_r - 0.001:
                efficient.append(p)
                max_r = max(max_r, p["ret"])

        result = {
            "tickers":    valid_tickers,
            "optWeights": opt_w.tolist(),
            "optReturn":  float(opt_ret),
            "optVol":     float(opt_vol),
            "optSharpe":  float(opt_sharpe),
            "eqReturn":   float(eq_ret),
            "eqVol":      float(eq_vol),
            "eqSharpe":   float(eq_sharpe),
            "frontier":   efficient,
            "mu":         mean_returns.tolist(),
            "sigma":      [float(np.sqrt(cov_matrix.iloc[i, i])) for i in range(n)],
            "cached":     False,
        }

        set_cached(ck, result)
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
