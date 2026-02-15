import io
from datetime import date, timedelta
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize

# Fallback ticker list if S&P 500 fetch fails
POPULAR_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "UNH",
    "JNJ", "V", "PG", "JPM", "MA", "HD", "DIS", "BAC", "ADBE", "NFLX",
    "XOM", "CVX", "WMT", "LLY", "AVGO", "COST", "MRK", "ABBV", "PEP", "TMO",
    "CSCO", "ACN", "MCD", "ABT", "CRM", "DHR", "VZ", "ADP", "WFC", "LIN",
    "NKE", "BMY", "PM", "TXN", "RTX", "QCOM", "UPS", "HON", "AMGN", "DE",
]


@st.cache_data(ttl=86400)
def load_sp500_tickers() -> List[str]:
    """Fetch S&P 500 constituents from GitHub. Fallback to POPULAR_TICKERS on failure."""
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df = pd.read_csv(url)
        tickers = sorted(df["Symbol"].astype(str).str.upper().unique().tolist())
        return tickers if tickers else POPULAR_TICKERS
    except Exception:
        return POPULAR_TICKERS


st.set_page_config(
    page_title="Build Your Smart Investment Portfolio",
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
    min_vol_point: Dict[str, float],
    max_ret_point: Dict[str, float],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        frontier["target_volatility"],
        frontier["target_return"],
        color="#1f77b4",
        linewidth=2,
    )
    # Minimum Risk Portfolio (Low Risk)
    ax.scatter(
        min_vol_point["volatility"],
        min_vol_point["return"],
        marker="o",
        color="#2ca02c",
        s=180,
        zorder=5,
        edgecolors="darkgreen",
        linewidths=2,
    )
    ax.annotate(
        "Minimum Risk\n(Low Risk)",
        (min_vol_point["volatility"], min_vol_point["return"]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=9,
        color="#2ca02c",
    )
    # Maximum Sharpe Portfolio (Balanced)
    ax.scatter(
        tan_point["volatility"],
        tan_point["return"],
        marker="*",
        color="#ff7f0e",
        s=280,
        zorder=5,
        edgecolors="darkorange",
        linewidths=2,
    )
    ax.annotate(
        "Best Risk–Return\n(Balanced)",
        (tan_point["volatility"], tan_point["return"]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=9,
        color="#ff7f0e",
    )
    # High Return portfolio
    ax.scatter(
        max_ret_point["volatility"],
        max_ret_point["return"],
        marker="^",
        color="#d62728",
        s=180,
        zorder=5,
        edgecolors="darkred",
        linewidths=2,
    )
    ax.annotate(
        "High Return",
        (max_ret_point["volatility"], max_ret_point["return"]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=9,
        color="#d62728",
    )
    if tan_point["volatility"] > 0:
        max_sigma = max(frontier["target_volatility"].max(), tan_point["volatility"]) * 1.1
        sigma_range = np.linspace(0, max_sigma, 25)
        cml = risk_free_rate + (tan_point["return"] - risk_free_rate) / tan_point["volatility"] * sigma_range
        ax.plot(sigma_range, cml, linestyle="--", color="#2ca02c", linewidth=1.5, alpha=0.8)

    ax.set_title("Risk vs Return: Choose Your Mix")
    ax.set_xlabel("Risk (Volatility)")
    ax.set_ylabel("Expected Return")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="#2ca02c", marker="o", linestyle="", label="Low Risk"),
            plt.Line2D([0], [0], color="#ff7f0e", marker="*", linestyle="", label="Balanced"),
            plt.Line2D([0], [0], color="#d62728", marker="^", linestyle="", label="High Return"),
        ],
        loc="lower right",
    )
    ax.grid(True, linestyle=":", linewidth=0.7)
    fig.tight_layout()
    return fig


st.title("Build Your Smart Investment Portfolio")
st.markdown("*Find the best risk–return mix for your chosen stocks.*")
st.divider()

TIME_PRESETS = {
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365,
    "5 Years": 365 * 5,
    "10 Years": 365 * 10,
    "20 Years": 365 * 20,
}

col_left, col_right = st.columns([2, 3])
frontier_plot = None  # Will be set when computation succeeds

with col_left:
    TICKER_OPTIONS = load_sp500_tickers()
    selected_tickers = st.multiselect(
        "Stock tickers",
        options=TICKER_OPTIONS,
        default=["AAPL", "MSFT", "GOOG"],
        placeholder="Type to search...",
        max_selections=20,
    )
    ticker_input = ", ".join(selected_tickers)

    time_choice = st.radio(
        "Time horizon",
        options=list(TIME_PRESETS.keys()),
        index=4,
        horizontal=True,
    )
    today = date.today()
    start_date = today - timedelta(days=TIME_PRESETS[time_choice])
    end_date = today

    risk_free_rate_input = st.slider(
        "Risk-free rate (% per year)",
        min_value=-5.0,
        max_value=20.0,
        value=2.0,
        step=0.5,
        help="Return from a safe investment (e.g. FD or government bond).",
    ) / 100

    frequency_map = {
        "Daily (252 trading days)": 252,
        "Weekly (52 weeks)": 52,
        "Monthly (12 months)": 12,
    }
    period_label = st.selectbox(
        "Return frequency",
        list(frequency_map.keys()),
        index=0,
        help="How often we treat returns: daily (most data), weekly, or monthly.",
    )
    periods_per_year = frequency_map[period_label]

    allow_short_sales = st.toggle("Allow short selling", value=False)

    frontier_plot = None
    can_compute = (
        selected_tickers
        and len(selected_tickers) >= 2
        and start_date < end_date
    )

    if can_compute:
        try:
            prices_df = load_price_data_from_yfinance(ticker_input, start_date, end_date)
            if prices_df.shape[1] < 2:
                st.error("Please ensure there are at least two assets with valid price data.")
            else:
                requested_tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
                missing_tickers = sorted(set(requested_tickers) - set(prices_df.columns))
                if missing_tickers:
                    st.warning("No data found for: " + ", ".join(missing_tickers))
                mean_returns, cov_matrix = compute_annual_metrics(prices_df, periods_per_year)
                num_assets = len(mean_returns)
                bounds = tuple(
                    (-1.0, 1.0) if allow_short_sales else (0.0, 1.0)
                    for _ in range(num_assets)
                )
                target_returns = np.linspace(mean_returns.min(), mean_returns.max(), num=50)
                frontier_df = build_efficient_frontier(
                    mean_returns, cov_matrix, target_returns, risk_free_rate_input, bounds
                )
                tangency_weights = max_sharpe_weights(
                    mean_returns, cov_matrix, risk_free_rate_input, bounds
                )
                tan_ret = portfolio_return(tangency_weights, mean_returns)
                tan_vol = portfolio_volatility(tangency_weights, cov_matrix)
                tan_sharpe = (tan_ret - risk_free_rate_input) / tan_vol if tan_vol > 0 else np.nan
                min_vol_row = frontier_df.loc[frontier_df["target_volatility"].idxmin()]
                min_vol_point = {
                    "return": min_vol_row["target_return"],
                    "volatility": min_vol_row["target_volatility"],
                }
                max_ret_row = frontier_df.loc[frontier_df["target_return"].idxmax()]
                max_ret_point = {
                    "return": max_ret_row["target_return"],
                    "volatility": max_ret_row["target_volatility"],
                }
                cml_data = {
                    "return": tan_ret,
                    "volatility": tan_vol,
                    "sharpe": tan_sharpe,
                }
                frontier_plot = draw_frontier_plot(
                    frontier_df, cml_data, risk_free_rate_input, min_vol_point, max_ret_point
                )

                st.subheader("What This Means For You")
                example_amount = 100_000
                vol_range = frontier_df["target_volatility"].max() - frontier_df["target_volatility"].min()
                tan_vol_pct = (tan_vol - frontier_df["target_volatility"].min()) / vol_range if vol_range > 0 else 0.5
                if tan_vol_pct < 0.33:
                    risk_label = "Low"
                    risk_explanation = "This portfolio has relatively low volatility. Good if you prefer stability."
                elif tan_vol_pct < 0.66:
                    risk_label = "Medium"
                    risk_explanation = "Moderate risk with a balanced reward. A common choice for many investors."
                else:
                    risk_label = "High"
                    risk_explanation = "Higher risk and higher expected return. Suitable if you can tolerate swings."

                st.metric("Expected return (annual)", f"{tan_ret:.1%}")
                st.caption("Average return you might expect per year based on history.")
                st.metric("Risk level", risk_label)
                st.caption(risk_explanation)
                st.write("**Suggested allocation (Balanced portfolio)**")
                for asset, w in zip(mean_returns.index, tangency_weights):
                    st.write(f"- **{asset}**: {w:.0%}")
                st.write("**If you invest Rs 1,00,000:**")
                for asset, w in zip(mean_returns.index, tangency_weights):
                    amt = example_amount * w
                    st.write(f"- **{asset}**: Rs {amt:,.0f}")

                with st.expander("Preview: historical price data", expanded=False):
                    st.dataframe(prices_df.tail())
        except Exception as exc:
            st.error(f"Unable to load price data: {exc}")

with col_right:
    if not selected_tickers or len(selected_tickers) < 2:
        st.info("Select at least 2 tickers to see the efficient frontier chart.")
    elif frontier_plot is not None:
        st.pyplot(frontier_plot)
        with st.expander("Technical Details (For Advanced Users)", expanded=False):
            st.write("**Efficient frontier data** (return, volatility, Sharpe ratio for each efficient portfolio):")
            frontier_display = frontier_df[["target_return", "target_volatility", "sharpe_ratio"]].copy()
            frontier_display.columns = ["Expected Return", "Volatility", "Sharpe Ratio"]
            st.dataframe(
                frontier_display.style.format(
                    {
                        "Expected Return": "{:.2%}",
                        "Volatility": "{:.2%}",
                        "Sharpe Ratio": "{:.2f}",
                    }
                )
            )
            st.write("**Recommended portfolio (Maximum Sharpe / Tangency)** — weights that maximize return per unit of risk:")
            weights_df = pd.DataFrame(
                {
                    "Asset": mean_returns.index,
                    "Weight": tangency_weights,
                }
            )
            st.dataframe(weights_df.set_index("Asset").style.format({"Weight": "{:.2%}"}))
            st.caption(
                f"Expected return: {tan_ret:.2%}, Volatility: {tan_vol:.2%}, Sharpe ratio: {tan_sharpe:.2f}"
            )
            st.write("**Covariance matrix** (how asset returns move together; used in optimization):")
            st.dataframe(cov_matrix.style.format("{:.4f}"))
            st.caption(
                "The app finds portfolio weights that minimize risk for each target return (efficient frontier) "
                "and the portfolio that maximizes the Sharpe ratio (tangency portfolio), using scipy.optimize."
            )
            download_df = frontier_df.drop(columns="weights")
            csv_buffer = io.StringIO()
            download_df.to_csv(csv_buffer, index=False)
            st.download_button(
                "Download frontier data (CSV)",
                data=csv_buffer.getvalue(),
                file_name="efficient_frontier.csv",
                mime="text/csv",
                key="download_frontier",
            )
    else:
        st.info("Data could not be loaded. Check your tickers and try again.")
