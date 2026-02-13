import io
from datetime import date, timedelta
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize


st.set_page_config(
    page_title="Efficient Frontier & CML",
    page_icon="💹",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_price_data(file) -> pd.DataFrame:
    """Read uploaded CSV into a DataFrame indexed by date if possible."""
    df = pd.read_csv(file)
    if df.empty:
        raise ValueError("Uploaded CSV is empty.")

    # Try to use the first column as a datetime index when possible.
    first_column = df.columns[0]
    try:
        df[first_column] = pd.to_datetime(df[first_column])
        df = df.set_index(first_column)
    except (ValueError, TypeError):
        df.index.name = "Index"

    # Drop non-numeric columns (e.g., tickers duplicated in headers).
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("CSV must contain numeric price columns.")

    return numeric_df


@st.cache_data(show_spinner=True)
def load_price_data_from_yfinance(
    tickers: str, start_date: date, end_date: date
) -> pd.DataFrame:
    """Download adjusted close prices for given tickers and date range using yfinance."""
    tickers_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not tickers_list:
        raise ValueError("Please enter at least one ticker symbol.")

    data = yf.download(
        tickers=tickers_list,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        raise ValueError("No data returned. Check tickers and date range.")

    # yfinance usually returns a MultiIndex with price fields on level 0
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" not in data.columns.levels[0]:
            raise ValueError("Downloaded data does not contain 'Adj Close' prices.")
        prices = data["Adj Close"].copy()
    else:
        # Single ticker case may return a simple DataFrame
        if "Adj Close" not in data.columns:
            raise ValueError("Downloaded data does not contain 'Adj Close' prices.")
        prices = data[["Adj Close"]].copy()
        prices.columns = tickers_list[:1]

    # Drop assets with all NaNs and dates with all NaNs
    prices = prices.dropna(axis=1, how="all").dropna(how="all")

    if prices.empty:
        raise ValueError(
            "No valid adjusted close price data found. "
            "Tickers may be invalid or have no data in the selected range."
        )

    return prices


def compute_annual_metrics(
    prices: pd.DataFrame, periods_per_year: int
) -> Tuple[pd.Series, pd.DataFrame]:
    """Convert price history to annualized mean returns and covariance."""
    returns = prices.pct_change().dropna(how="any")
    if returns.empty:
        raise ValueError("Not enough observations to compute returns.")

    mean_returns = returns.mean() * periods_per_year
    cov_matrix = returns.cov() * periods_per_year
    return mean_returns, cov_matrix


def portfolio_return(weights: np.ndarray, mean_returns: pd.Series) -> float:
    return float(np.dot(weights, mean_returns))


def portfolio_volatility(weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
    return float(np.sqrt(np.dot(weights.T, cov_matrix @ weights)))


def min_variance_weights(
    target_return: float,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    bounds: Tuple[Tuple[float, float], ...],
) -> np.ndarray:
    num_assets = len(mean_returns)
    init_guess = np.repeat(1.0 / num_assets, num_assets)

    def objective(weights: np.ndarray) -> float:
        return portfolio_volatility(weights, cov_matrix)

    constraints = (
        {"type": "eq", "fun": lambda x: np.sum(x) - 1.0},
        {"type": "eq", "fun": lambda x: portfolio_return(x, mean_returns) - target_return},
    )

    result = minimize(
        objective,
        init_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise ValueError(f"Efficient frontier optimization failed: {result.message}")

    return result.x


def max_sharpe_weights(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float,
    bounds: Tuple[Tuple[float, float], ...],
) -> np.ndarray:
    num_assets = len(mean_returns)
    init_guess = np.repeat(1.0 / num_assets, num_assets)

    def neg_sharpe(weights: np.ndarray) -> float:
        excess_ret = portfolio_return(weights, mean_returns) - risk_free_rate
        vol = portfolio_volatility(weights, cov_matrix)
        if vol == 0:
            return np.inf
        return -excess_ret / vol

    constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1.0},)

    result = minimize(
        neg_sharpe,
        init_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise ValueError(f"Max Sharpe optimization failed: {result.message}")

    return result.x


def build_efficient_frontier(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    target_returns: np.ndarray,
    risk_free_rate: float,
    bounds: Tuple[Tuple[float, float], ...],
) -> pd.DataFrame:
    rows = []
    for target in target_returns:
        weights = min_variance_weights(target, mean_returns, cov_matrix, bounds)
        ret = portfolio_return(weights, mean_returns)
        vol = portfolio_volatility(weights, cov_matrix)
        sharpe = (ret - risk_free_rate) / vol if vol > 0 else np.nan
        rows.append(
            {
                "target_return": ret,
                "target_volatility": vol,
                "sharpe_ratio": sharpe,
                "weights": weights,
            }
        )
    return pd.DataFrame(rows)


def draw_frontier_plot(
    frontier: pd.DataFrame,
    tan_point: Dict[str, float],
    risk_free_rate: float,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        frontier["target_volatility"],
        frontier["target_return"],
        label="Efficient Frontier",
        color="#1f77b4",
    )
    ax.scatter(
        tan_point["volatility"],
        tan_point["return"],
        marker="*",
        color="#ff7f0e",
        s=200,
        label="Tangency Portfolio",
    )
    if tan_point["volatility"] > 0:
        max_sigma = max(frontier["target_volatility"].max(), tan_point["volatility"]) * 1.1
        sigma_range = np.linspace(0, max_sigma, 25)
        cml = risk_free_rate + (tan_point["return"] - risk_free_rate) / tan_point["volatility"] * sigma_range
        ax.plot(sigma_range, cml, linestyle="--", color="#2ca02c", label="Capital Market Line")

    ax.set_title("Efficient Frontier & Capital Market Line")
    ax.set_xlabel("Volatility (σ)")
    ax.set_ylabel("Expected Return")
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.7)
    fig.tight_layout()
    return fig


st.title("Efficient Frontier & Capital Market Line")
st.write(
    "Enter stock ticker symbols and a date range to download historical adjusted close "
    "prices using yfinance, then model the Efficient Frontier and Capital Market Line."
)

ticker_input = st.text_input(
    "Ticker symbols (comma-separated)", value="AAPL, MSFT, GOOG"
)

today = date.today()
default_start = today - timedelta(days=365 * 5)
col_start, col_end = st.columns(2)
start_date = col_start.date_input("Start date", value=default_start)
end_date = col_end.date_input("End date", value=today)

risk_free_rate_input = st.number_input(
    "Risk-free rate (% annualized)",
    min_value=-5.0,
    max_value=20.0,
    value=2.0,
    step=0.1,
) / 100

frequency_map = {
    "Daily (252 trading days)": 252,
    "Weekly (52 weeks)": 52,
    "Monthly (12 months)": 12,
}
period_label = st.selectbox("Return frequency", list(frequency_map.keys()), index=0)
periods_per_year = frequency_map[period_label]

allow_short_sales = st.toggle("Allow short selling", value=False)

if not ticker_input.strip():
    st.info("Enter at least one ticker symbol to begin.")
    st.stop()

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

try:
    prices_df = load_price_data_from_yfinance(ticker_input, start_date, end_date)
except Exception as exc:
    st.error(f"Unable to download price data: {exc}")
    st.stop()

if prices_df.shape[1] < 2:
    st.error("Please ensure there are at least two assets with valid price data.")
    st.stop()

requested_tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
missing_tickers = sorted(set(requested_tickers) - set(prices_df.columns))
if missing_tickers:
    st.warning(
        "No data found for the following tickers in the selected date range: "
        + ", ".join(missing_tickers)
    )

st.subheader("Price Preview")
st.dataframe(prices_df.tail())

try:
    mean_returns, cov_matrix = compute_annual_metrics(prices_df, periods_per_year)
except Exception as exc:
    st.error(f"Failed to compute statistics: {exc}")
    st.stop()

num_assets = len(mean_returns)
bounds = tuple(
    (-1.0, 1.0) if allow_short_sales else (0.0, 1.0)
    for _ in range(num_assets)
)

target_returns = np.linspace(mean_returns.min(), mean_returns.max(), num=50)
try:
    frontier_df = build_efficient_frontier(
        mean_returns, cov_matrix, target_returns, risk_free_rate_input, bounds
    )
except Exception as exc:
    st.error(exc)
    st.stop()

try:
    tangency_weights = max_sharpe_weights(mean_returns, cov_matrix, risk_free_rate_input, bounds)
except Exception as exc:
    st.error(exc)
    st.stop()

tan_ret = portfolio_return(tangency_weights, mean_returns)
tan_vol = portfolio_volatility(tangency_weights, cov_matrix)
tan_sharpe = (tan_ret - risk_free_rate_input) / tan_vol if tan_vol > 0 else np.nan

cml_data = {
    "return": tan_ret,
    "volatility": tan_vol,
    "sharpe": tan_sharpe,
}

frontier_plot = draw_frontier_plot(frontier_df, cml_data, risk_free_rate_input)
st.pyplot(frontier_plot)

st.subheader("Efficient Frontier Data")
frontier_display = frontier_df[["target_return", "target_volatility", "sharpe_ratio"]].copy()
st.dataframe(
    frontier_display.style.format(
        {
            "target_return": "{:.2%}",
            "target_volatility": "{:.2%}",
            "sharpe_ratio": "{:.2f}",
        }
    )
)

st.subheader("Tangency Portfolio (Max Sharpe)")
weights_df = pd.DataFrame(
    {
        "Asset": mean_returns.index,
        "Weight": tangency_weights,
    }
)
st.dataframe(weights_df.set_index("Asset").style.format({"Weight": "{:.2%}"}))

st.caption(
    f"Tangency portfolio expected return: {tan_ret:.2%}, volatility: {tan_vol:.2%}, "
    f"Sharpe ratio: {tan_sharpe:.2f}"
)

download_df = frontier_df.drop(columns="weights")
csv_buffer = io.StringIO()
download_df.to_csv(csv_buffer, index=False)

st.download_button(
    "Download frontier data (CSV)",
    data=csv_buffer.getvalue(),
    file_name="efficient_frontier.csv",
    mime="text/csv",
)
