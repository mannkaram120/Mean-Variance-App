from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
try:
    from sklearn.covariance import LedoitWolf
except ImportError:
    LedoitWolf = None
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import time
from urllib.request import urlopen
from xml.etree import ElementTree as ET

app = Flask(__name__)
CORS(app)

print("LedoitWolf available:", LedoitWolf is not None)

CACHE_TTL_SECONDS = 1800
OPTIMIZE_CACHE = {}
CACHE_LOCK = Lock()
RISK_FREE_CACHE_TTL_SECONDS = 21600
RISK_FREE_CACHE = None
STRESS_TEST_CACHE_TTL_SECONDS = 3600
STRESS_TEST_CACHE = {}
YF_DOWNLOAD_LOCK = Lock()  # serializes yf.download calls — prevents data mixing under concurrent requests

MARKET_CAP_CACHE = {}
MARKET_CAP_CACHE_TTL = 86400  # 24 hours

STRESS_SCENARIOS = {
    '2008 Global Financial Crisis': ('2008-06-01', '2009-03-31'),
    'COVID Crash': ('2020-02-20', '2020-08-31'),
    '2022 Rate Hike Cycle': ('2022-01-01', '2022-12-31'),
    'Dot-Com Bust': ('2000-03-01', '2002-10-31'),
}
STRESS_SCENARIO_CONTEXT = {
    '2008 Global Financial Crisis': 'Global credit markets seized up after the Lehman collapse. Financial contagion and forced deleveraging drove broad risk-off selling.',
    'COVID Crash': 'Pandemic lockdowns triggered a sudden growth scare, followed by aggressive fiscal and monetary support that powered a rapid recovery.',
    '2022 Rate Hike Cycle': 'Fed funds rate rose from 0.25% to 4.50% in 12 months. Growth stocks with high P/E multiples hit hardest.',
    'Dot-Com Bust': 'The post-bubble unwind punished speculative tech valuations as earnings expectations reset across the market.',
}


def covariance_condition_number(sigma):
    arr = np.asarray(sigma)
    eigs = np.linalg.eigvalsh(arr)
    min_eig = max(float(eigs[0]), 1e-12)
    return round(float(eigs[-1] / min_eig), 2)


def _instability_interpretation(mvo_score, bl_score, mvo_weights, bl_weights, tickers):

    mvo_weights = mvo_weights or {}
    bl_weights = bl_weights or {}

    mvo_max_w = max(mvo_weights.values()) if mvo_weights else 0.0
    bl_max_w = max(bl_weights.values()) if bl_weights else 0.0
    mvo_concentrated = mvo_max_w > 0.95
    bl_concentrated = bl_max_w > 0.95

    mvo_dominant = max(mvo_weights, key=mvo_weights.get) if mvo_concentrated else None
    bl_dominant = max(bl_weights, key=bl_weights.get) if bl_concentrated else None

    if mvo_concentrated and not bl_concentrated:
        return (
            f"MVO scores {mvo_score:.4f} because it allocated {mvo_max_w*100:.0f}% to "
            f"{mvo_dominant} — a corner solution pinned at a constraint boundary. "
            f"Perturbations cannot move weights that are already at the wall. "
            f"LW+BL scores {bl_score:.4f} because its diversified allocation has room to shift "
            f"under input changes, which reflects genuine sensitivity, not instability."
        )
    elif bl_concentrated and not mvo_concentrated:
        return (
            f"LW+BL scores {bl_score:.4f} due to a concentrated allocation in {bl_dominant}. "
            f"MVO's diversified weights are more sensitive to perturbations, scoring {mvo_score:.4f}."
        )
    elif mvo_score < bl_score:
        return (
            f"MVO scores lower ({mvo_score:.4f} vs {bl_score:.4f}). Both are interior solutions — "
            f"LW+BL weights shift more under perturbation, likely due to flatter Sharpe surface "
            f"from BL return shrinkage toward equilibrium."
        )
    else:
        return (
            f"LW+BL scores lower ({bl_score:.4f} vs {mvo_score:.4f}), consistent with "
            f"Ledoit-Wolf shrinkage improving covariance conditioning and stabilising weight solutions."
        )


def _get_treasury_1y_xml_url():
    return (
        'https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml'
        '?data=daily_treasury_yield_curve&field_tdr_date_value=' + str(date.today().year)
    )


def _find_latest_1y_treasury_rate():
    with urlopen(_get_treasury_1y_xml_url(), timeout=10) as resp:
        xml_bytes = resp.read()

    root = ET.fromstring(xml_bytes)
    latest_date = None
    latest_rate = None

    for entry in root.iter():
        if not entry.tag.endswith('entry'):
            continue

        entry_date = None
        entry_rate = None

        for node in entry.iter():
            tag = node.tag.split('}')[-1]
            text = (node.text or '').strip()
            if not text:
                continue
            if tag == 'NEW_DATE':
                entry_date = text
            elif tag == 'BC_1YEAR':
                try:
                    entry_rate = float(text)
                except ValueError:
                    entry_rate = None

        if entry_date and entry_rate is not None:
            if latest_date is None or entry_date > latest_date:
                latest_date = entry_date
                latest_rate = entry_rate

    if latest_rate is None:
        raise ValueError('Unable to extract latest 1-year Treasury yield')

    return {
        'rate_percent': round(latest_rate, 2),
        'rate_decimal': round(latest_rate / 100.0, 6),
        'source': 'U.S. Treasury Daily Par Yield Curve',
        'as_of': latest_date[:10] if latest_date else None
    }


def _extract_close_prices(raw, requested_tickers):
    if raw is None or raw.empty:
        raise ValueError('No price data returned. Check your tickers.')

    print(f"[EXTRACT] raw shape={raw.shape}, columns type={'MultiIndex' if isinstance(raw.columns, pd.MultiIndex) else 'flat'}, date range: {raw.index[0] if len(raw)>0 else 'EMPTY'} to {raw.index[-1] if len(raw)>0 else 'EMPTY'}")

    if len(requested_tickers) == 1:
        if 'Close' in raw.columns:
            prices = raw[['Close']].copy()
            prices.columns = requested_tickers
        else:
            raise ValueError('Close price not found for ' + requested_tickers[0])
    elif isinstance(raw.columns, pd.MultiIndex):
        level0_vals = list(set(raw.columns.get_level_values(0)))
        level1_vals = list(set(raw.columns.get_level_values(1)))
        print(f"[EXTRACT] MultiIndex level0 unique: {level0_vals[:6]}, level1 unique: {level1_vals[:6]}")
        if 'Close' in raw.columns.get_level_values(0):
            prices = raw['Close'].copy()
        elif 'Close' in raw.columns.get_level_values(1):
            prices = raw.xs('Close', axis=1, level=1).copy()
        else:
            raise ValueError('Could not extract Close prices')
    else:
        prices = raw.copy()

    prices = prices.dropna(axis=1, how='all').ffill().dropna()
    print(f"[EXTRACT] output shape={prices.shape}, columns={list(prices.columns)}")
    return prices


def _corr_payload_from_frame(frame):
    return {
        'labels': list(frame.columns),
        'matrix': [[round(float(v), 4) for v in row] for row in frame.values.tolist()]
    }


def _fetch_single_market_cap(ticker):
    try:
        info = yf.Ticker(ticker).info
        return ticker, info.get('marketCap', 0) or 0
    except Exception:
        return ticker, 0


def get_market_cap_weights(tickers):
    now_ts = time.time()
    caps = {}
    to_fetch = []
    for ticker in tickers:
        with CACHE_LOCK:
            cached = MARKET_CAP_CACHE.get(ticker)
        if cached and cached['expires_at'] > now_ts:
            caps[ticker] = cached['value']
        else:
            to_fetch.append(ticker)
    if to_fetch:
        with ThreadPoolExecutor(max_workers=min(len(to_fetch), 8)) as ex:
            results = list(ex.map(_fetch_single_market_cap, to_fetch))
        for ticker, cap in results:
            caps[ticker] = cap
            with CACHE_LOCK:
                MARKET_CAP_CACHE[ticker] = {
                    'value': cap,
                    'expires_at': now_ts + MARKET_CAP_CACHE_TTL
                }
    total = sum(caps.values())
    if total <= 0:
        return {t: 1.0 / len(tickers) for t in tickers}
    return {t: caps[t] / total for t in tickers}


def compute_black_litterman(mu_hist, sigma_lw, tickers,
                             views, rfr, delta=2.5, tau=0.05):
    """
    mu_hist  : pd.Series  — historical annualised returns
    sigma_lw : pd.DataFrame — LW annualised covariance
    tickers  : list of str
    views    : list of dicts, one of two formats:
                 Absolute view:
                   {"type": "absolute",    # or omit type field
                    "ticker": "AAPL",
                    "view_return": 0.134,  # expected absolute return
                    "confidence": 0.7}
                 Relative view:
                   {"type": "relative",
                    "long_ticker": "AAPL",
                    "short_ticker": "MSFT",
                    "view_return": 0.05,   # expected outperformance
                    "confidence": 0.7}
    rfr      : float
    Returns  : (mu_bl pd.Series, w_mkt_dict dict)
    """
    n = len(tickers)
    sigma_vals = sigma_lw.values

    w_mkt_dict = get_market_cap_weights(tickers)
    w_mkt = np.array([w_mkt_dict.get(t, 1.0 / n) for t in tickers])
    w_mkt = w_mkt / w_mkt.sum()  # normalise

    # Equilibrium returns: pi = delta * Sigma * w_mkt
    pi = delta * sigma_vals @ w_mkt

    # No views — return equilibrium directly
    if not views:
        return pd.Series(pi, index=tickers), w_mkt_dict

    ticker_idx = {t: i for i, t in enumerate(tickers)}
    valid_views = [v for v in views if (
        (v.get('type', 'absolute') == 'absolute'
         and v.get('ticker') in ticker_idx)
        or
        (v.get('type') == 'relative'
         and v.get('long_ticker') in ticker_idx
         and v.get('short_ticker') in ticker_idx
         and v.get('long_ticker') != v.get('short_ticker'))
    )]

    if not valid_views:
        return pd.Series(pi, index=tickers), w_mkt_dict

    k = len(valid_views)
    P = np.zeros((k, n))
    Q = np.zeros(k)
    omega_diag = np.zeros(k)

    for i, v in enumerate(valid_views):
        view_type = v.get('type', 'absolute')

        if view_type == 'absolute':
            idx = ticker_idx[v['ticker']]
            P[i, idx] = 1.0
            Q[i] = float(v['view_return'])

        elif view_type == 'relative':
            long_idx  = ticker_idx[v['long_ticker']]
            short_idx = ticker_idx[v['short_ticker']]
            P[i, long_idx]  =  1.0
            P[i, short_idx] = -1.0
            Q[i] = float(v['view_return'])

        conf = float(v.get('confidence', 0.5))
        conf = max(0.01, min(0.99, conf))
        omega_diag[i] = ((1 - conf) / conf) * tau * float(
            P[i] @ sigma_vals @ P[i])

    Omega = np.diag(omega_diag)

    # He & Litterman (2002) BL posterior mean
    tau_sigma = tau * sigma_vals
    try:
        tau_sigma_inv = np.linalg.inv(tau_sigma)
        omega_inv = np.linalg.inv(Omega)
        M = np.linalg.inv(tau_sigma_inv + P.T @ omega_inv @ P)
        mu_bl = M @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)
    except np.linalg.LinAlgError:
        mu_bl = pi  # fallback to equilibrium

    return pd.Series(mu_bl, index=tickers), w_mkt_dict


def compute_instability_score(mu_vals, sigma_vals, w_base,
                               bounds, sum_con, rfr, n_perturb=20):
    """Perturb mu slightly, measure average weight change. Lower = more stable."""
    import math
    weight_changes = []
    for _ in range(n_perturb):
        noise = np.random.normal(0, 0.01, len(mu_vals))
        mu_p = mu_vals + noise
        def neg_sr(w, _mu=mu_p, _s=sigma_vals, _r=rfr):
            v = math.sqrt(max(float(w @ _s @ w), 0.0))
            return -(float(w @ _mu) - _r) / v if v > 1e-10 else 1e10
        res = minimize(neg_sr, w_base.copy(), method='SLSQP',
                       bounds=bounds, constraints=[sum_con],
                       options={'maxiter': 200, 'ftol': 1e-6})
        if res.success:
            weight_changes.append(float(np.sum(np.abs(res.x - w_base))))
    return float(np.mean(weight_changes)) if weight_changes else 0.0


def _fetch_cached_prices(tickers, start_date, end_date, cache_prefix, cache_scope=None):
    # Sort tickers so cache hits regardless of input order
    cache_key = (cache_prefix, cache_scope, tuple(sorted(tickers)), start_date, end_date)
    now_ts = time.time()

    with CACHE_LOCK:
        cached_entry = STRESS_TEST_CACHE.get(cache_key)
        if cached_entry and cached_entry['expires_at'] > now_ts:
            print(f"[CACHE HIT] {cache_prefix}/{cache_scope} {start_date}-{end_date} ({len(cached_entry['prices'])} rows)")
            return cached_entry['prices'].copy()

    with YF_DOWNLOAD_LOCK:
        # Double-check cache after acquiring lock — another thread may have populated it
        with CACHE_LOCK:
            cached_entry = STRESS_TEST_CACHE.get(cache_key)
            if cached_entry and cached_entry['expires_at'] > now_ts:
                print(f"[CACHE HIT after lock] {cache_prefix}/{cache_scope} {start_date}-{end_date}")
                return cached_entry['prices'].copy()

        # Download each ticker INDIVIDUALLY to prevent yfinance batch-download data
        # contamination. yf.download() with multiple tickers shares internal session
        # state that produces wrong date ranges when concurrent threads download
        # different periods. Individual yf.Ticker().history() calls are fully isolated.
        print(f"[YF DOWNLOAD] {cache_prefix}/{cache_scope} tickers={tickers} {start_date}-{end_date}")
        end_for_yf = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        close_series = {}
        for ticker in tickers:
            for attempt in range(2):  # Two attempts per ticker
                try:
                    hist = yf.Ticker(ticker).history(
                        start=start_date, end=end_for_yf, auto_adjust=True, timeout=30
                    )
                    if hist is not None and not hist.empty and 'Close' in hist.columns:
                        close_series[ticker] = hist['Close']
                        print(f"[YF]   {ticker}: {len(hist)} rows, "
                              f"{hist.index[0].strftime('%Y-%m-%d')} to {hist.index[-1].strftime('%Y-%m-%d')}")
                        break  # Success, move to next ticker
                    else:
                        col_info = list(hist.columns) if hist is not None and not hist.empty else 'empty'
                        if attempt == 0:
                            print(f"[YF]   {ticker}: no Close data (cols={col_info}), retrying...")
                            time.sleep(1)  # Brief delay before retry
                        else:
                            print(f"[YF]   {ticker}: no Close data (cols={col_info})")
                except Exception as e:
                    if attempt == 0:
                        print(f"[YF]   {ticker}: ERROR {e}, retrying...")
                        time.sleep(1)
                    else:
                        print(f"[YF]   {ticker}: ERROR {e}")

        if not close_series:
            raise ValueError('No price data returned. Check your tickers.')

        prices = pd.DataFrame(close_series)
        # yf.Ticker().history() returns tz-aware DatetimeIndex — strip timezone
        if prices.index.tz is not None:
            prices.index = prices.index.tz_localize(None)
        prices = prices.dropna(axis=1, how='all').ffill().dropna()

        print(f"[YF RESULT] {cache_scope}: {len(prices)} rows, "
              f"{prices.index[0] if len(prices) > 0 else 'EMPTY'} to "
              f"{prices.index[-1] if len(prices) > 0 else 'EMPTY'}, "
              f"cols={list(prices.columns)}")

        with CACHE_LOCK:
            STRESS_TEST_CACHE[cache_key] = {
                'prices': prices.copy(),
                'expires_at': time.time() + STRESS_TEST_CACHE_TTL_SECONDS
            }

    return prices


@app.route('/risk-free-rate', methods=['GET'])
def risk_free_rate():
    global RISK_FREE_CACHE
    now_ts = time.time()

    with CACHE_LOCK:
        if RISK_FREE_CACHE and RISK_FREE_CACHE['expires_at'] > now_ts:
            return jsonify(RISK_FREE_CACHE['payload'])

    try:
        payload = _find_latest_1y_treasury_rate()
    except Exception:
        payload = {
            'rate_percent': 2.00,
            'rate_decimal': 0.02,
            'source': 'Fallback default',
            'as_of': date.today().isoformat(),
            'live': False
        }
    else:
        payload['live'] = True

    with CACHE_LOCK:
        RISK_FREE_CACHE = {
            'payload': payload,
            'expires_at': now_ts + RISK_FREE_CACHE_TTL_SECONDS
        }

    return jsonify(payload)

@app.route('/', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'Portfolio Optimizer API running'})


@app.route('/stress-test', methods=['POST'])
def stress_test():
    try:
        body = request.get_json(force=True)
        tickers = [str(t).strip().upper() for t in body.get('tickers', [])]
        weights = body.get('weights', {}) or {}
        scenario_name = str(body.get('scenario', '')).strip()

        print(f"[STRESS-TEST] scenario='{scenario_name}' tickers={tickers}")

        if len(tickers) < 2:
            return jsonify({'error': 'Please select at least 2 tickers'}), 400
        if scenario_name not in STRESS_SCENARIOS:
            print(f"[STRESS-TEST] ERROR: '{scenario_name}' not in STRESS_SCENARIOS keys: {list(STRESS_SCENARIOS.keys())}")
            return jsonify({'error': 'Unknown stress test scenario'}), 400

        start_date, end_date = STRESS_SCENARIOS[scenario_name]
        print(f"[STRESS-TEST] date range: {start_date} to {end_date}")
        scenario_tickers = list(dict.fromkeys(tickers + ['SPY']))
        prices = _fetch_cached_prices(scenario_tickers, start_date, end_date, 'stress-test', scenario_name)

        # Sanity check: verify the returned data actually covers the requested period.
        # If yfinance returned wrong-period data (known concurrency bug), purge and retry.
        if len(prices) > 0:
            actual_start = prices.index[0].strftime('%Y-%m-%d')
            actual_end = prices.index[-1].strftime('%Y-%m-%d')
            expected_year = start_date[:4]
            actual_year = actual_start[:4]
            if actual_year != expected_year:
                print(f"[STRESS-TEST] DATE MISMATCH! Expected year {expected_year}, got {actual_year} (actual={actual_start}). Purging cache and retrying.")
                cache_key = ('stress-test', scenario_name, tuple(sorted(scenario_tickers)), start_date, end_date)
                with CACHE_LOCK:
                    STRESS_TEST_CACHE.pop(cache_key, None)
                prices = _fetch_cached_prices(scenario_tickers, start_date, end_date, 'stress-test', scenario_name)
                # Second check after retry
                if len(prices) > 0:
                    retry_year = prices.index[0].strftime('%Y')
                    if retry_year != expected_year:
                        print(f"[STRESS-TEST] STILL WRONG after retry! Expected {expected_year}, got {retry_year}. Returning error.")
                        return jsonify({
                            'error': f'Data date mismatch: expected year {expected_year} but yfinance returned {retry_year}. Please try again.',
                            'scenario': scenario_name,
                            'start_date': start_date,
                            'end_date': end_date
                        }), 500
        print(f"[STRESS-TEST] prices shape={prices.shape}, date range: {prices.index[0]} to {prices.index[-1]}, columns={list(prices.columns)}")

        available_columns = list(prices.columns)
        available_assets = [ticker for ticker in tickers if ticker in available_columns]
        warnings = []

        missing_assets = [ticker for ticker in tickers if ticker not in available_assets]

        if 'SPY' not in available_columns:
            return jsonify({'error': 'SPY benchmark has no data for this scenario.'}), 400

        if len(available_assets) < 2:
            return jsonify({
                'scenario': scenario_name,
                'start_date': start_date,
                'end_date': end_date,
                'warnings': warnings,
                'error': 'Insufficient data for this stress test scenario.'
            }), 200

        original_weight_sum = sum(float(weights.get(ticker, 0.0)) for ticker in tickers)
        available_weight_sum = sum(float(weights.get(ticker, 0.0)) for ticker in available_assets)
        optimized_run_exists = original_weight_sum > 1e-12

        if optimized_run_exists and missing_assets and available_weight_sum > 1e-12:
            redistributed_weights = {
                ticker: float(weights.get(ticker, 0.0)) / available_weight_sum for ticker in available_assets
            }
            weight_mode = 'redistributed'
            excluded_text = ', '.join(missing_assets)
            warnings = [
                '⚠ ' + excluded_text + ' has no data for this scenario and has been excluded from stress test. '
                + 'Weights redistributed proportionally among available tickers.'
            ]
        elif optimized_run_exists and missing_assets and available_weight_sum <= 1e-12:
            redistributed_weights = {ticker: 1.0 / len(available_assets) for ticker in available_assets}
            weight_mode = 'equal_weight_fallback'
            excluded_text = ', '.join(missing_assets)
            warnings = [
                '⚠ ' + excluded_text + ' has no data for this scenario. '
                + 'Your optimized weights cannot be applied to the remaining assets. '
                + 'Showing equal weight fallback across available tickers - this does not represent your optimized portfolio.'
            ]
        elif optimized_run_exists:
            redistributed_weights = {
                ticker: float(weights.get(ticker, 0.0)) for ticker in available_assets
            }
            weight_mode = 'optimized'
        else:
            redistributed_weights = {ticker: 1.0 / len(available_assets) for ticker in available_assets}
            weight_mode = 'equal_weight_fallback'
            warnings = ['⚠ Run optimization first to use optimized weights. Showing equal weight fallback for this scenario.']

        # Fix dropna: only require SPY to be non-NaN so missing asset data in early
        # periods (e.g. GOOG didn't exist in 2000) doesn't collapse the whole dataset
        prices = prices[available_assets + ['SPY']].ffill()
        daily_returns = prices.pct_change()
        daily_returns = daily_returns.dropna(subset=['SPY'])
        # Fill remaining NaNs in asset columns with 0 (treat as no return on missing days)
        daily_returns[available_assets] = daily_returns[available_assets].fillna(0.0)

        if daily_returns.empty or len(daily_returns) < 5:
            return jsonify({
                'scenario': scenario_name,
                'start_date': start_date,
                'end_date': end_date,
                'warnings': warnings,
                'error': 'Insufficient data for this stress test scenario.'
            }), 200

        ew_n = len(available_assets)
        ew_weights = np.repeat(1.0 / ew_n, ew_n)
        opt_weight_vector = np.array([redistributed_weights[ticker] for ticker in available_assets], dtype=float)
        optimized_daily = daily_returns[available_assets] @ opt_weight_vector
        equal_daily = daily_returns[available_assets] @ ew_weights
        spy_daily = daily_returns['SPY']

        optimized_cum = (1.0 + optimized_daily).cumprod()
        equal_cum = (1.0 + equal_daily).cumprod()
        spy_cum = (1.0 + spy_daily).cumprod()
        asset_cum = (1.0 + daily_returns[available_assets]).cumprod()

        optimized_cum = optimized_cum / optimized_cum.iloc[0] * 100.0
        equal_cum = equal_cum / equal_cum.iloc[0] * 100.0
        spy_cum = spy_cum / spy_cum.iloc[0] * 100.0
        asset_cum = asset_cum / asset_cum.iloc[0] * 100.0

        opt_rolling_max = optimized_cum.cummax()
        opt_drawdown_series = (optimized_cum - opt_rolling_max) / opt_rolling_max
        eq_rolling_max = equal_cum.cummax()
        eq_drawdown_series = (equal_cum - eq_rolling_max) / eq_rolling_max
        opt_drawdown = abs(float(opt_drawdown_series.min())) * 100.0
        opt_total_return = (optimized_cum.iloc[-1] / 100.0 - 1.0) * 100.0
        eq_total_return = (equal_cum.iloc[-1] / 100.0 - 1.0) * 100.0
        spy_total_return = (spy_cum.iloc[-1] / 100.0 - 1.0) * 100.0
        versus_spy = opt_total_return - spy_total_return
        spy_rolling_max = spy_cum.cummax()
        spy_drawdown_series = (spy_cum - spy_rolling_max) / spy_rolling_max
        spy_drawdown = abs(float(spy_drawdown_series.min())) * 100.0
        drawdown_diff = spy_drawdown - opt_drawdown
        opt_crisis_vol = float(optimized_daily.std() * np.sqrt(252) * 100.0)
        spy_crisis_vol = float(spy_daily.std() * np.sqrt(252) * 100.0)
        crisis_vol_diff = spy_crisis_vol - opt_crisis_vol
        asset_total_returns = (
            prices.iloc[-1][available_assets] / prices.iloc[0][available_assets] - 1.0
        ) * 100.0
        asset_contributions = {
            ticker: float(redistributed_weights[ticker] * asset_total_returns[ticker])
            for ticker in available_assets
        }
        top_weight = max(redistributed_weights.values()) if redistributed_weights else 0.0
        concentration_note = (
            'Single-asset concentration typically shows higher volatility than diversified benchmarks during crisis periods. '
            'Diversification across uncorrelated assets reduces crisis volatility.'
            if top_weight >= 0.8 else
            'Crisis volatility reflects how concentrated the portfolio remained versus the diversified SPY benchmark.'
        )
        drawdown_note = (
            'Drawdown measures worst intra-period loss. Total return measures start-to-end performance. '
            'A portfolio can underperform on drawdown but outperform on total return if recovery is faster.'
        )
        recovery_note = (
            'Optimized portfolio returned '
            + f'{opt_total_return:.2f}%'
            + ' over the full period vs SPY\'s '
            + f'{spy_total_return:.2f}%'
            + ', '
            + ('outperforming' if versus_spy >= 0 else 'underperforming')
            + ' by '
            + f'{abs(versus_spy):.2f}%. '
            + 'Peak drawdown reached '
            + f'{opt_drawdown:.2f}%'
            + " vs SPY's "
            + f'{spy_drawdown:.2f}%'
            + '.'
        )

        def _series_payload(series):
            return {
                'dates': [idx.strftime('%Y-%m-%d') for idx in series.index],
                'values': [round(float(v), 4) for v in series.values]
            }

        opt_series = _series_payload(optimized_cum)
        spy_series = _series_payload(spy_cum)
        print(f"[STRESS-TEST RESPONSE] scenario='{scenario_name}' first_date={opt_series['dates'][0] if opt_series['dates'] else 'EMPTY'} last_date={opt_series['dates'][-1] if opt_series['dates'] else 'EMPTY'} drawdown={round(float(opt_drawdown),2)}")

        resp = make_response(jsonify({
            'scenario': scenario_name,
            'scenario_context': STRESS_SCENARIO_CONTEXT.get(scenario_name, ''),
            'start_date': start_date,
            'end_date': end_date,
            'warnings': warnings,
            'weight_mode': weight_mode,
            'optimized_run_exists': optimized_run_exists,
            'available_tickers': available_assets,
            'redistributed_weights': {k: round(v, 6) for k, v in redistributed_weights.items()},
            'asset_stats': {
                ticker: {
                    'crisis_return': round(float(asset_total_returns[ticker]), 2),
                    'contribution': round(float(asset_contributions[ticker]), 2)
                }
                for ticker in available_assets
            },
            'optimized': opt_series,
            'equal_weight': _series_payload(equal_cum),
            'spy': spy_series,
            'drawdowns': {
                'optimized': _series_payload(opt_drawdown_series * 100.0),
                'equal_weight': _series_payload(eq_drawdown_series * 100.0),
                'spy': _series_payload(spy_drawdown_series * 100.0),
            },
            'metrics': {
                'max_drawdown': {
                    'optimized': round(float(opt_drawdown), 2),
                    'spy': round(float(spy_drawdown), 2),
                    'difference': round(float(drawdown_diff), 2)
                },
                'total_return': {
                    'optimized': round(float(opt_total_return), 2),
                    'equal_weight': round(float(eq_total_return), 2),
                    'spy': round(float(spy_total_return), 2),
                    'difference': round(float(versus_spy), 2)
                },
                'crisis_volatility': {
                    'optimized': round(opt_crisis_vol, 2),
                    'spy': round(spy_crisis_vol, 2),
                    'difference': round(crisis_vol_diff, 2)
                },
                'recovery_note': recovery_note,
                'drawdown_note': drawdown_note,
                'concentration_note': concentration_note
            }
        }))
        resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        return resp

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'detail': traceback.format_exc()}), 500

@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        body        = request.get_json(force=True)
        tickers     = [str(t).strip().upper() for t in body.get('tickers', [])]
        days        = int(body.get('days', 365))
        rfr         = float(body.get('risk_free_rate', 0.02))
        frequency   = int(body.get('frequency', 252))
        allow_short = bool(body.get('allow_short', False))
        views = body.get('views', []) or []

        if len(tickers) < 2:
            return jsonify({'error': 'Please select at least 2 tickers'}), 400

        end_date   = date.today()
        start_date = end_date - timedelta(days=days)
        cache_key = (
            tuple(tickers),
            days,
            round(rfr, 6),
            frequency,
            allow_short,
            end_date.isoformat(),
            str(sorted([(v.get('ticker',''), v.get('view_return',0), v.get('confidence',0.5)) for v in views]))
        )

        now_ts = time.time()
        with CACHE_LOCK:
            cached_entry = OPTIMIZE_CACHE.get(cache_key)
            if cached_entry and cached_entry['expires_at'] > now_ts:
                print("CACHE HIT — returning cached result (bl:", cached_entry['payload'].get('bl') is not None, ")")
                return jsonify(cached_entry['payload'])
            expired_keys = [k for k, v in OPTIMIZE_CACHE.items() if v['expires_at'] <= now_ts]
            for k in expired_keys:
                del OPTIMIZE_CACHE[k]

        prices = _fetch_cached_prices(tickers, start_date.isoformat(), end_date.isoformat(), 'optimize')
        valid_tickers = list(prices.columns)

        if len(valid_tickers) < 2:
            return jsonify({'error': 'Not enough valid tickers with data'}), 400

        returns  = prices.pct_change().dropna()
        if len(returns) < 20:
            return jsonify({'error': 'Too few trading days. Try a longer time horizon.'}), 400

        # Annualised stats
        mu    = returns.mean() * frequency
        sigma = returns.cov()  * frequency
        cond_sample = covariance_condition_number(sigma.values)
        n     = len(valid_tickers)

        w0     = np.repeat(1.0 / n, n)
        bounds = tuple((-1.0, 1.0) if allow_short else (0.0, 1.0) for _ in range(n))
        sum_con = {'type': 'eq', 'fun': lambda w: float(np.sum(w)) - 1.0}

        def port_ret(w): return float(w @ mu.values)
        def port_vol(w): return float(np.sqrt(w @ sigma.values @ w))
        def neg_sharpe(w):
            v = port_vol(w)
            return -(port_ret(w) - rfr) / v if v > 1e-10 else 1e10
        def solve_max_sharpe_for_sigma(sigma_matrix, start_weights):
            sigma_vals_local = sigma_matrix.values
            def port_vol_sigma(w): return float(np.sqrt(w @ sigma_vals_local @ w))
            def neg_sharpe_sigma(w):
                v = port_vol_sigma(w)
                return -(float(w @ mu.values) - rfr) / v if v > 1e-10 else 1e10
            res = minimize(neg_sharpe_sigma, start_weights, method='SLSQP', bounds=bounds,
                           constraints=[sum_con], options={'maxiter':1000,'ftol':1e-9})
            if not res.success:
                return None
            w = res.x
            ret = float(w @ mu.values)
            vol = float(np.sqrt(w @ sigma_vals_local @ w))
            sr = (ret - rfr) / vol if vol > 0 else 0.0
            return {'weights': w, 'ret': ret, 'vol': vol, 'sr': sr}

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
        if allow_short:
            def neg_ret(w): return -port_ret(w)
            res_maxret = minimize(neg_ret, w0, method='SLSQP', bounds=bounds,
                                  constraints=[sum_con], options={'maxiter':1000,'ftol':1e-9})
            if res_maxret.success:
                w_maxret = res_maxret.x
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
        targets = list(np.linspace(ret_lo, ret_hi, 120))

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

        # ── Monte Carlo: 10,000 random portfolios — fully vectorised (no loop)
        N_SIM = 10000
        if not allow_short:
            rnd = np.random.dirichlet(np.ones(n), size=N_SIM)
        else:
            batches = []
            accepted = 0
            batch_size = max(20000, N_SIM * 2)
            while accepted < N_SIM:
                trial = np.random.uniform(-1.0, 1.0, size=(batch_size, n - 1))
                last_col = 1.0 - trial.sum(axis=1, keepdims=True)
                valid = np.abs(last_col[:, 0]) <= 1.0
                if np.any(valid):
                    batch = np.hstack([trial[valid], last_col[valid]])
                    batches.append(batch)
                    accepted += batch.shape[0]
            rnd = np.vstack(batches)[:N_SIM]
        # Portfolio returns: (N_SIM,)
        mc_ret = rnd @ mu_vals
        # Portfolio vols: sqrt of (w @ sigma @ w) for each row — vectorised
        mc_vol = np.sqrt(np.einsum('ij,jk,ik->i', rnd, sigma_vals, rnd))
        mc_sr  = (mc_ret - rfr) / np.where(mc_vol > 1e-10, mc_vol, 1e-10)

        # Downsample to 2000 points for payload size — keep distribution shape
        idx     = np.random.choice(N_SIM, size=min(2000, N_SIM), replace=False)
        mc_cloud = [
            {'vol': round(float(mc_vol[i]),5),
             'ret': round(float(mc_ret[i]),5),
             'sr':  round(float(mc_sr[i]), 4)}
            for i in idx
        ]

        def w_dict(w): return {t: round(float(x),4) for t,x in zip(valid_tickers,w)}

        returns_subset = returns[valid_tickers]
        corr_matrix = returns_subset.corr()
        corr_payload = _corr_payload_from_frame(corr_matrix)
        lw_corr_payload = None
        lw_portfolio_payload = None
        lw_cov = None
        lw_shrinkage = None
        cond_lw = None
        if LedoitWolf is not None:
            try:
                lw = LedoitWolf().fit(returns_subset.values)
                lw_cov = pd.DataFrame(lw.covariance_ * frequency, index=valid_tickers, columns=valid_tickers)
                print("LW fitted. lw_cov shape:", lw_cov.shape, "shrinkage:", round(float(lw.shrinkage_), 4))
                cond_lw = covariance_condition_number(lw_cov.values)
                lw_diag = np.sqrt(np.clip(np.diag(lw_cov.values), 1e-12, None))
                lw_corr = lw_cov.div(lw_diag, axis=0).div(lw_diag, axis=1)
                lw_corr_payload = _corr_payload_from_frame(lw_corr)
                lw_shrinkage = round(float(lw.shrinkage_), 4)
                lw_sharpe = solve_max_sharpe_for_sigma(lw_cov, w_sharpe.copy())
                if lw_sharpe is not None:
                    lw_portfolio_payload = {
                        'ret': round(lw_sharpe['ret'], 6),
                        'vol': round(lw_sharpe['vol'], 6),
                        'sr': round(lw_sharpe['sr'], 4),
                        'weights': w_dict(lw_sharpe['weights'])
                    }
            except Exception as lw_err:
                print("LW FAILED:", lw_err)
                import traceback; traceback.print_exc()

        # ── Black-Litterman enhanced optimization
        bl_payload = None
        bl_frontier_payload = None

        if LedoitWolf is not None and lw_cov is not None:
            print("BL block entered. lw_cov shape:", lw_cov.shape)
            try:
                mu_bl, w_mkt_dict = compute_black_litterman(
                    mu, lw_cov, valid_tickers, views, rfr)

                sigma_bl = lw_cov.values
                mu_bl_vals = mu_bl.values

                def port_ret_bl(w): return float(w @ mu_bl_vals)
                def port_vol_bl(w): return float(np.sqrt(w @ sigma_bl @ w))
                def neg_sharpe_bl(w):
                    v = port_vol_bl(w)
                    return -(port_ret_bl(w) - rfr) / v if v > 1e-10 else 1e10

                res_bl = minimize(neg_sharpe_bl, w0, method='SLSQP',
                                  bounds=bounds, constraints=[sum_con],
                                  options={'maxiter': 1000, 'ftol': 1e-9})

                print("BL mu_bl:", mu_bl.values.tolist())
                print("BL res_bl.success:", res_bl.success, "| message:", res_bl.message)

                if res_bl.success:
                    w_bl = res_bl.x
                    ret_bl = port_ret_bl(w_bl)
                    vol_bl = port_vol_bl(w_bl)
                    sr_bl = (ret_bl - rfr) / vol_bl if vol_bl > 0 else 0.0

                    # BL Min Variance
                    res_bl_minv = minimize(
                        port_vol_bl, w0, method='SLSQP',
                        bounds=bounds, constraints=[sum_con],
                        options={'maxiter': 1000, 'ftol': 1e-9})
                    w_bl_minv = res_bl_minv.x if res_bl_minv.success else w0
                    ret_bl_minv = port_ret_bl(w_bl_minv)
                    vol_bl_minv = port_vol_bl(w_bl_minv)

                    # BL Frontier — 120 points parallel
                    bl_targets = list(np.linspace(
                        ret_bl_minv, float(mu_bl.max()), 120))

                    def compute_bl_point(target):
                        ef = minimize(
                            port_vol_bl, w_bl_minv.copy(),
                            method='SLSQP', bounds=bounds,
                            constraints=[sum_con,
                                {'type': 'eq',
                                 'fun': lambda w, t=target: float(w @ mu_bl_vals) - t}],
                            options={'maxiter': 100, 'ftol': 1e-6})
                        if ef.success:
                            v = float(np.sqrt(ef.x @ sigma_bl @ ef.x))
                            r = float(ef.x @ mu_bl_vals)
                            return {'vol': round(v, 6), 'ret': round(r, 6)}
                        return None

                    with ThreadPoolExecutor(max_workers=8) as ex:
                        bl_raw = list(ex.map(compute_bl_point, bl_targets))
                    bl_frontier_payload = sorted(
                        [r for r in bl_raw if r is not None],
                        key=lambda x: x['ret'])

                    # BL Daily VaR
                    d_ret_bl = float(w_bl @ returns_subset.mean().values)
                    d_vol_bl = float(np.sqrt(w_bl @ lw_cov.values @ w_bl))
                    var_95_bl = float(-(d_ret_bl - 1.645 * d_vol_bl))

                    # Equilibrium returns for display
                    w_mkt_arr = np.array([w_mkt_dict.get(t, 1.0/n)
                                           for t in valid_tickers])
                    w_mkt_arr = w_mkt_arr / w_mkt_arr.sum()
                    pi_vals = 2.5 * lw_cov.values @ w_mkt_arr

                    bl_payload = {
                        'expected_return': round(ret_bl, 6),
                        'volatility': round(vol_bl, 6),
                        'sharpe_ratio': round(sr_bl, 4),
                        'var_95': round(var_95_bl, 6),
                        'weights': w_dict(w_bl),
                        'equilibrium_returns': {
                            t: round(float(v), 6)
                            for t, v in zip(valid_tickers, pi_vals)},
                        'bl_returns': {
                            t: round(float(v), 6)
                            for t, v in mu_bl.items()},
                        'market_cap_weights': {
                            t: round(float(v), 6)
                            for t, v in w_mkt_dict.items()},
                        'views_applied': len(views),
                        'min_variance': {
                            'ret': round(ret_bl_minv, 6),
                            'vol': round(vol_bl_minv, 6),
                            'weights': w_dict(w_bl_minv)
                        }
                    }
            except Exception as bl_err:
                bl_payload = None
                bl_frontier_payload = None
                print("BL FAILED:", str(bl_err))
                import traceback; traceback.print_exc()
                _bl_error = str(bl_err)

        # ── Instability scores
        instability_payload = None
        try:
            mvo_instab = compute_instability_score(
                mu_vals, sigma_vals, w_sharpe, bounds, sum_con, rfr, n_perturb=50)
            bl_instab = None
            if bl_payload is not None and 'mu_bl' in locals():
                bl_w_arr = np.array(
                    [float(bl_payload['weights'].get(t, 0)) for t in valid_tickers])
                bl_instab = compute_instability_score(
                    mu_bl.values, lw_cov.values, bl_w_arr,
                    bounds, sum_con, rfr, n_perturb=50)
            instability_payload = {
                'mvo': round(mvo_instab, 4),
                'bl': round(bl_instab, 4) if bl_instab is not None else None,
                'cond_sample': cond_sample,
                'cond_lw': cond_lw
            }
        except Exception:
            instability_payload = None

        response_payload = {
            # Primary result (Max Sharpe)
            'expected_return': round(ret_sharpe, 6),
            'volatility':      round(vol_sharpe, 6),
            'sharpe_ratio':    round(sr_sharpe,  4),
            'var_95':          round(var_95,      6),
            'var_95_label':    'Daily VaR (95%)',
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
            'mc_cloud':     mc_cloud,
            'correlation_matrix': corr_payload,
            'raw_correlation': corr_payload,
            'lw_correlation': lw_corr_payload,
            'lw_portfolio': lw_portfolio_payload,
            'tickers_used': valid_tickers,
            'simulated':    False,
            'bl': bl_payload,
            'bl_frontier': bl_frontier_payload,
            'mu_hist': {t: round(float(v), 6) for t, v in mu.items()},
            'lw_shrinkage': lw_shrinkage,
            'instability': instability_payload,
            'bl_error': _bl_error if '_bl_error' in dir() else None,
        }

        if instability_payload and bl_payload and instability_payload.get('bl') is not None:
            instability_payload['caption'] = _instability_interpretation(
                instability_payload['mvo'],
                instability_payload['bl'],
                response_payload['weights'],
                bl_payload['weights'],
                valid_tickers
            )

        with CACHE_LOCK:
            OPTIMIZE_CACHE[cache_key] = {
                'payload': response_payload,
                'expires_at': now_ts + CACHE_TTL_SECONDS
            }

        return jsonify(response_payload)

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'detail': traceback.format_exc()}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
