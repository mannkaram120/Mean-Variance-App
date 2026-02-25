from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from datetime import date, timedelta

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'Portfolio Optimizer API running'})

@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        body        = request.get_json(force=True)
        tickers     = [str(t).strip().upper() for t in body.get('tickers', [])]
        days        = int(body.get('days', 365))
        rfr         = float(body.get('risk_free_rate', 0.02))
        frequency   = int(body.get('frequency', 252))
        allow_short = bool(body.get('allow_short', False))

        if len(tickers) < 2:
            return jsonify({'error': 'Please select at least 2 tickers'}), 400

        end_date   = date.today()
        start_date = end_date - timedelta(days=days)

        # Download with group_by='ticker' to get consistent structure
        raw = yf.download(
            tickers=' '.join(tickers),
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            auto_adjust=True,
            progress=False,
            group_by='ticker'
        )

        if raw is None or raw.empty:
            return jsonify({'error': 'No price data returned. Check your tickers.'}), 400

        # ── Extract Close prices robustly for both single and multi-ticker downloads
        if len(tickers) == 1:
            # Single ticker: flat columns
            if 'Close' in raw.columns:
                prices = raw[['Close']].copy()
                prices.columns = tickers
            else:
                return jsonify({'error': 'Close price not found for ' + tickers[0]}), 400
        elif isinstance(raw.columns, pd.MultiIndex):
            # Multi-ticker with MultiIndex: (field, ticker)
            if 'Close' in raw.columns.get_level_values(0):
                prices = raw['Close'].copy()
            elif 'Close' in raw.columns.get_level_values(1):
                prices = raw.xs('Close', axis=1, level=1).copy()
            else:
                return jsonify({'error': 'Could not extract Close prices from data'}), 400
        else:
            # Flat columns — ticker symbols as column names
            prices = raw.copy()

        # Drop all-NaN columns and rows
        prices = prices.dropna(axis=1, how='all').dropna(how='all')

        # Forward-fill small gaps then drop remaining NaNs
        prices = prices.ffill().dropna()

        valid_tickers = list(prices.columns)

        if len(valid_tickers) < 2:
            bad = list(set(tickers) - set(valid_tickers))
            return jsonify({
                'error': 'Not enough valid data. Could not fetch: ' + ', '.join(bad) if bad else 'Not enough data for optimisation.'
            }), 400

        # ── Compute annualised returns & covariance
        returns = prices.pct_change().dropna()

        if len(returns) < 20:
            return jsonify({'error': 'Too few trading days in selected period. Try a longer time horizon.'}), 400

        mu    = returns.mean() * frequency
        sigma = returns.cov()  * frequency

        n      = len(valid_tickers)
        w0     = np.repeat(1.0 / n, n)
        bounds = tuple((-1.0, 1.0) if allow_short else (0.0, 1.0) for _ in range(n))
        sum_constraint = {'type': 'eq', 'fun': lambda w: float(np.sum(w)) - 1.0}

        # ── Max Sharpe optimisation
        def neg_sharpe(w):
            ret = float(w @ mu.values)
            vol = float(np.sqrt(w @ sigma.values @ w))
            return -(ret - rfr) / vol if vol > 1e-10 else 1e10

        res = minimize(neg_sharpe, w0, method='SLSQP',
                       bounds=bounds,
                       constraints=[sum_constraint],
                       options={'maxiter': 1000, 'ftol': 1e-9})

        if not res.success:
            return jsonify({'error': 'Max Sharpe optimisation failed: ' + str(res.message)}), 500

        opt_w   = res.x
        opt_ret = float(opt_w @ mu.values)
        opt_vol = float(np.sqrt(opt_w @ sigma.values @ opt_w))
        opt_sr  = (opt_ret - rfr) / opt_vol if opt_vol > 0 else 0.0

        # ── VaR 95% (parametric, 1-day)
        daily_mu  = returns.mean().values
        daily_cov = returns.cov().values
        d_ret = float(opt_w @ daily_mu)
        d_vol = float(np.sqrt(opt_w @ daily_cov @ opt_w))
        var_95 = float(-(d_ret - 1.645 * d_vol))

        # ── Efficient frontier (30 points, min-variance for each target return)
        frontier = []
        ret_min = float(mu.min())
        ret_max = float(mu.max())
        for target in np.linspace(ret_min, ret_max, 30):
            def port_vol(w):
                return float(np.sqrt(w @ sigma.values @ w))
            ef_res = minimize(
                port_vol, w0, method='SLSQP',
                bounds=bounds,
                constraints=[
                    sum_constraint,
                    {'type': 'eq', 'fun': lambda w, t=target: float(w @ mu.values) - t}
                ],
                options={'maxiter': 500, 'ftol': 1e-8}
            )
            if ef_res.success:
                v = float(np.sqrt(ef_res.x @ sigma.values @ ef_res.x))
                r = float(ef_res.x @ mu.values)
                frontier.append({'vol': round(v, 6), 'ret': round(r, 6)})

        return jsonify({
            'expected_return': round(opt_ret, 6),
            'volatility':      round(opt_vol, 6),
            'sharpe_ratio':    round(opt_sr,  4),
            'var_95':          round(var_95,  6),
            'weights':         {t: round(float(w), 4) for t, w in zip(valid_tickers, opt_w)},
            'frontier':        frontier,
            'tickers_used':    valid_tickers,
            'days_used':       days,
            'simulated':       False,
            'rfr':             rfr
        })

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'detail': traceback.format_exc()}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
