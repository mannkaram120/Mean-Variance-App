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
        tickers     = body.get('tickers', [])
        days        = int(body.get('days', 365))
        rfr         = float(body.get('risk_free_rate', 0.02))
        frequency   = int(body.get('frequency', 252))
        allow_short = bool(body.get('allow_short', False))

        if len(tickers) < 2:
            return jsonify({'error': 'Need at least 2 tickers'}), 400

        end_date   = date.today()
        start_date = end_date - timedelta(days=days)

        raw = yf.download(
            tickers=tickers,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            auto_adjust=True,
            progress=False
        )

        if raw.empty:
            return jsonify({'error': 'No data returned from yFinance. Check tickers.'}), 400

        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw['Close']
        else:
            prices = raw[['Close']] if 'Close' in raw.columns else raw

        prices = prices.dropna(axis=1, how='all').dropna(how='all')

        if prices.shape[1] < 2:
            return jsonify({'error': 'Not enough valid tickers with data'}), 400

        valid_tickers = list(prices.columns)
        returns = prices.pct_change().dropna()
        mu      = returns.mean() * frequency
        sigma   = returns.cov()  * frequency

        n      = len(valid_tickers)
        w0     = np.repeat(1.0 / n, n)
        bounds = tuple((-1.0, 1.0) if allow_short else (0.0, 1.0) for _ in range(n))

        def neg_sharpe(w):
            ret = float(np.dot(w, mu))
            vol = float(np.sqrt(w @ sigma.values @ w))
            return -(ret - rfr) / vol if vol > 1e-10 else 1e10

        res = minimize(neg_sharpe, w0, method='SLSQP',
                       bounds=bounds,
                       constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}],
                       options={'maxiter': 1000, 'ftol': 1e-9})

        if not res.success:
            return jsonify({'error': 'Optimisation failed: ' + res.message}), 500

        opt_w   = res.x
        opt_ret = float(np.dot(opt_w, mu))
        opt_vol = float(np.sqrt(opt_w @ sigma.values @ opt_w))
        opt_sr  = (opt_ret - rfr) / opt_vol if opt_vol > 0 else 0

        daily_ret = float(np.dot(opt_w, returns.mean()))
        daily_vol = float(np.sqrt(opt_w @ returns.cov().values @ opt_w))
        var_95    = -(daily_ret - 1.645 * daily_vol)

        frontier = []
        ret_range = np.linspace(float(mu.min()), float(mu.max()), 30)
        for target in ret_range:
            def port_vol(w):
                return float(np.sqrt(w @ sigma.values @ w))
            ef_res = minimize(port_vol, w0, method='SLSQP',
                              bounds=bounds,
                              constraints=[
                                  {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                                  {'type': 'eq', 'fun': lambda w, t=target: float(np.dot(w, mu)) - t}
                              ],
                              options={'maxiter': 500})
            if ef_res.success:
                v = float(np.sqrt(ef_res.x @ sigma.values @ ef_res.x))
                r = float(np.dot(ef_res.x, mu))
                frontier.append({'vol': v, 'ret': r})

        return jsonify({
            'expected_return': opt_ret,
            'volatility':      opt_vol,
            'sharpe_ratio':    opt_sr,
            'var_95':          var_95,
            'weights':         {t: float(w) for t, w in zip(valid_tickers, opt_w)},
            'frontier':        frontier,
            'tickers_used':    valid_tickers,
            'simulated':       False
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
