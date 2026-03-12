from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor

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

        raw = yf.download(
            tickers=' '.join(tickers),
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            auto_adjust=True,
            progress=False,
            group_by='ticker',
            threads=True
        )

        if raw is None or raw.empty:
            return jsonify({'error': 'No price data returned. Check your tickers.'}), 400

        # Extract Close prices robustly
        if len(tickers) == 1:
            if 'Close' in raw.columns:
                prices = raw[['Close']].copy()
                prices.columns = tickers
            else:
                return jsonify({'error': 'Close price not found for ' + tickers[0]}), 400
        elif isinstance(raw.columns, pd.MultiIndex):
            if 'Close' in raw.columns.get_level_values(0):
                prices = raw['Close'].copy()
            elif 'Close' in raw.columns.get_level_values(1):
                prices = raw.xs('Close', axis=1, level=1).copy()
            else:
                return jsonify({'error': 'Could not extract Close prices'}), 400
        else:
            prices = raw.copy()

        prices = prices.dropna(axis=1, how='all').ffill().dropna()
        valid_tickers = list(prices.columns)

        if len(valid_tickers) < 2:
            return jsonify({'error': 'Not enough valid tickers with data'}), 400

        returns  = prices.pct_change().dropna()
        if len(returns) < 20:
            return jsonify({'error': 'Too few trading days. Try a longer time horizon.'}), 400

        # Annualised stats
        mu    = returns.mean() * frequency
        sigma = returns.cov()  * frequency
        n     = len(valid_tickers)

        w0     = np.repeat(1.0 / n, n)
        bounds = tuple((-1.0, 1.0) if allow_short else (0.0, 1.0) for _ in range(n))
        sum_con = {'type': 'eq', 'fun': lambda w: float(np.sum(w)) - 1.0}

        def port_ret(w): return float(w @ mu.values)
        def port_vol(w): return float(np.sqrt(w @ sigma.values @ w))
        def neg_sharpe(w):
            v = port_vol(w)
            return -(port_ret(w) - rfr) / v if v > 1e-10 else 1e10

        # ── Max Sharpe
        res_sharpe = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds,
                              constraints=[sum_con], options={'maxiter':1000,'ftol':1e-9})
        if not res_sharpe.success:
            return jsonify({'error': 'Max Sharpe failed: ' + str(res_sharpe.message)}), 500

        w_sharpe   = res_sharpe.x
        ret_sharpe = port_ret(w_sharpe)
        vol_sharpe = port_vol(w_sharpe)
        sr_sharpe  = (ret_sharpe - rfr) / vol_sharpe

        # ── Min Variance
        res_minv = minimize(port_vol, w0, method='SLSQP', bounds=bounds,
                            constraints=[sum_con], options={'maxiter':1000,'ftol':1e-9})
        if not res_minv.success:
            return jsonify({'error': 'Min Variance failed: ' + str(res_minv.message)}), 500

        w_minv   = res_minv.x
        ret_minv = port_ret(w_minv)
        vol_minv = port_vol(w_minv)
        sr_minv  = (ret_minv - rfr) / vol_minv

        # ── Max Return (just max weight on highest-mu asset)
        max_mu_idx = int(np.argmax(mu.values))
        w_maxret   = np.zeros(n)
        w_maxret[max_mu_idx] = 1.0
        if not allow_short:
            pass  # already valid
        ret_maxret = port_ret(w_maxret)
        vol_maxret = port_vol(w_maxret)
        sr_maxret  = (ret_maxret - rfr) / vol_maxret if vol_maxret > 0 else 0.0

        # ── Equal Weighted
        w_eq    = w0.copy()
        ret_eq  = port_ret(w_eq)
        vol_eq  = port_vol(w_eq)
        sr_eq   = (ret_eq - rfr) / vol_eq if vol_eq > 0 else 0.0

        # ── VaR 95% for optimal (Max Sharpe)
        d_ret  = float(w_sharpe @ returns.mean().values)
        d_vol  = float(np.sqrt(w_sharpe @ returns.cov().values @ w_sharpe))
        var_95 = float(-(d_ret - 1.645 * d_vol))

        # ── Efficient Frontier: 40 points computed in parallel
        ret_lo = ret_minv
        ret_hi = float(mu.max())
        targets = list(np.linspace(ret_lo, ret_hi, 40))

        sigma_vals = sigma.values
        mu_vals    = mu.values

        def compute_point(target):
            def _vol(w): return float(np.sqrt(w @ sigma_vals @ w))
            ef = minimize(_vol, w_minv.copy(), method='SLSQP', bounds=bounds,
                          constraints=[sum_con,
                                       {'type':'eq','fun': lambda w,t=target: float(w @ mu_vals)-t}],
                          options={'maxiter':100,'ftol':1e-6})
            if ef.success:
                v = float(np.sqrt(ef.x @ sigma_vals @ ef.x))
                r = float(ef.x @ mu_vals)
                return {'vol': round(v,6), 'ret': round(r,6), '_t': target}
            return None

        with ThreadPoolExecutor(max_workers=8) as ex:
            raw_results = list(ex.map(compute_point, targets))

        # Filter, sort by return to ensure smooth curve
        frontier = sorted(
            [r for r in raw_results if r is not None],
            key=lambda x: x['ret']
        )
        frontier = [{'vol': p['vol'], 'ret': p['ret']} for p in frontier]

        # Build weights dicts
        def w_dict(w): return {t: round(float(x),4) for t,x in zip(valid_tickers,w)}

        return jsonify({
            # Primary result (Max Sharpe)
            'expected_return': round(ret_sharpe, 6),
            'volatility':      round(vol_sharpe, 6),
            'sharpe_ratio':    round(sr_sharpe,  4),
            'var_95':          round(var_95,      6),
            'weights':         w_dict(w_sharpe),
            'rfr':             rfr,

            # Special portfolios
            'max_sharpe': {
                'ret': round(ret_sharpe,6), 'vol': round(vol_sharpe,6),
                'sr':  round(sr_sharpe,4),  'weights': w_dict(w_sharpe)
            },
            'min_variance': {
                'ret': round(ret_minv,6), 'vol': round(vol_minv,6),
                'sr':  round(sr_minv,4),  'weights': w_dict(w_minv)
            },
            'max_return': {
                'ret': round(ret_maxret,6), 'vol': round(vol_maxret,6),
                'sr':  round(sr_maxret,4),  'weights': w_dict(w_maxret)
            },
            'equal_weighted': {
                'ret': round(ret_eq,6), 'vol': round(vol_eq,6),
                'sr':  round(sr_eq,4),  'weights': w_dict(w_eq)
            },

            'frontier':     frontier,
            'tickers_used': valid_tickers,
            'simulated':    False
        })

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'detail': traceback.format_exc()}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
