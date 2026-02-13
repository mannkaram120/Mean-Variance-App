import io
from datetime import date, timedelta
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize

# Popular US stock tickers for autocomplete
POPULAR_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "UNH",
    "JNJ", "V", "PG", "JPM", "MA", "HD", "DIS", "BAC", "ADBE", "NFLX",
    "XOM", "CVX", "WMT", "LLY", "AVGO", "COST", "MRK", "ABBV", "PEP", "TMO",
    "CSCO", "ACN", "MCD", "ABT", "CRM", "DHR", "VZ", "ADP", "WFC", "LIN",
    "NKE", "BMY", "PM", "TXN", "RTX", "QCOM", "UPS", "HON", "AMGN", "DE",
    "LOW", "INTU", "SPGI", "AMAT", "GE", "BKNG", "AXP", "SBUX", "GILD", "ADI",
    "ISRG", "TJX", "C", "BLK", "MDT", "VRTX", "ZTS", "REGN", "CME", "SCHW",
    "AAL", "AAP", "ABBV", "ABC", "ABMD", "ABT", "ACGL", "ACHC", "ACI", "ACMR",
    "ADBE", "ADM", "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", "A", "AIG",
    "AIZ", "AJG", "AKAM", "ALB", "ALGN", "ALK", "ALL", "ALLE", "AMAT", "AMCR",
    "AMD", "AME", "AMGN", "AMP", "AMT", "AMZN", "ANET", "ANSS", "ANTM", "AON",
    "AOS", "APA", "APD", "APH", "APTV", "ARE", "ATO", "ATVI", "AVB", "AVGO",
    "AVY", "AWK", "AXP", "AZO", "BA", "BAC", "BALL", "BAX", "BBWI", "BBY",
    "BDX", "BEN", "BF.B", "BIIB", "BIO", "BK", "BKNG", "BKR", "BLK", "BLL",
    "BMY", "BR", "BRK.B", "BSX", "BWA", "BXP", "C", "CAG", "CAH", "CAM",
    "CAT", "CB", "CBOE", "CBRE", "CCI", "CCL", "CDAY", "CDNS", "CDW", "CE",
    "CF", "CFG", "CHD", "CHRW", "CHTR", "CI", "CINF", "CL", "CLH", "CLX",
    "CMA", "CMCSA", "CME", "CMG", "CMI", "CMS", "CNC", "CNP", "COF", "COO",
    "COP", "COST", "CPB", "CPRT", "CPT", "CRL", "CRM", "CSCO", "CSGP", "CSX",
    "CTAS", "CTLT", "CTRA", "CTSH", "CTVA", "CUBE", "CURI", "CVS", "CVX", "CZR",
    "D", "DAL", "DD", "DDOG", "DE", "DFS", "DG", "DGX", "DHI", "DHR",
    "DIS", "DISCA", "DISH", "DLR", "DLTR", "DOC", "DOCN", "DOCU", "DOV", "DOW",
    "DPZ", "DRE", "DRI", "DT", "DUK", "DVA", "DVN", "DXCM", "DXPE", "EA",
    "EBAY", "ECL", "ED", "EFX", "EG", "EIX", "EL", "ELAN", "EMN", "EMR",
    "ENPH", "ENTG", "EOG", "EPAM", "EQH", "EQIX", "EQR", "EQT", "ES", "ESS",
    "ETN", "ETR", "EVRG", "EW", "EXAS", "EXC", "EXPD", "EXPE", "EXR", "F",
    "FANG", "FAST", "FBHS", "FCX", "FDS", "FDX", "FE", "FFIV", "FERG", "FIS",
    "FISV", "FITB", "FIVE", "FIVN", "FLT", "FMC", "FNF", "FOX", "FOXA", "FRC",
    "FRT", "FSLR", "FTNT", "FTV", "FWRD", "G", "GATX", "GD", "GE", "GEHC",
    "GEN", "GILD", "GIS", "GL", "GLW", "GM", "GNRC", "GOOG", "GOOGL", "GPC",
    "GPN", "GRMN", "GS", "GTLS", "HAL", "HAS", "HBAN", "HCA", "HD", "HES",
    "HIG", "HII", "HLT", "HOLX", "HON", "HPE", "HPQ", "HRL", "HSIC", "HST",
    "HSY", "HUBB", "HUM", "HWM", "HXL", "IBM", "ICE", "IDXX", "IEX", "IFF",
    "ILMN", "INCY", "INFO", "INTC", "INTU", "INVH", "IP", "IPG", "IQV", "IR",
    "IRM", "ISRG", "IT", "ITT", "ITW", "IVZ", "J", "JBHT", "JBL", "JCI",
    "JKHY", "JNJ", "JNPR", "JPM", "K", "KDP", "KHC", "KIM", "KLAC", "KMB",
    "KMI", "KMX", "KO", "KR", "KRC", "KRX", "KSS", "KVUE", "L", "LAMR",
    "LBRDK", "LBRT", "LDOS", "LEN", "LEVI", "LFC", "LH", "LHX", "LIN", "LKQ",
    "LLY", "LMT", "LNC", "LNT", "LOW", "LPLA", "LRCX", "LSI", "LULU", "LUV",
    "LVS", "LW", "LYB", "LYV", "MA", "MAA", "MANH", "MAR", "MAS", "MASI",
    "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT", "MELI", "MET", "META", "MGM",
    "MHK", "MKC", "MKTX", "MLI", "MLM", "MMC", "MMM", "MNST", "MO", "MOH",
    "MOS", "MPC", "MPWR", "MRK", "MRNA", "MRO", "MRVL", "MS", "MSCI", "MSFT",
    "MSI", "MTB", "MTCH", "MTD", "MU", "MUR", "NCLH", "NDAQ", "NDSN", "NEE",
    "NEM", "NFLX", "NI", "NKE", "NOC", "NOV", "NOW", "NRG", "NSC", "NTAP",
    "NTRS", "NUE", "NVDA", "NVR", "NWS", "NWSA", "NXST", "O", "ODFL", "OGN",
    "OKE", "OMC", "ON", "ONON", "ORCL", "ORLY", "OTIS", "OXY", "PANW", "PARA",
    "PAYC", "PAYX", "PCAR", "PCG", "PCH", "PDD", "PEAK", "PEG", "PENN", "PEP",
    "PFE", "PG", "PGR", "PH", "PHM", "PKG", "PKI", "PLD", "PM", "PNC",
    "PNR", "PNW", "POOL", "POR", "PPG", "PPL", "PRGO", "PRU", "PSA", "PSX",
    "PTC", "PTON", "PWR", "PXD", "PYPL", "QCOM", "QRVO", "R", "RCL", "RE",
    "REG", "REGN", "RF", "RHI", "RJF", "RL", "RMD", "ROK", "ROL", "ROP",
    "ROST", "RPRX", "RSG", "RTX", "RVTY", "RYAN", "S", "SAIA", "SBAC", "SBUX",
    "SCHW", "SCI", "SEIC", "SHW", "SIRI", "SJM", "SLB", "SNA", "SNPS", "SO",
    "SPG", "SPGI", "SRE", "STE", "STLD", "STT", "STX", "STZ", "SWK", "SWKS",
    "SYF", "SYK", "SYY", "T", "TAP", "TDG", "TDY", "TECH", "TEL", "TER",
    "TFC", "TFX", "TGT", "TJX", "TKO", "TMO", "TMUS", "TPG", "TROW", "TRV",
    "TSCO", "TSLA", "TSN", "TT", "TTD", "TTWO", "TXN", "TXT", "TYL", "U",
    "UAL", "UBER", "UDR", "UHS", "ULTA", "UNH", "UNP", "UPS", "URI", "USB",
    "V", "VFC", "VICI", "VLO", "VMC", "VRSK", "VRSN", "VRTX", "VSAT", "VST",
    "VTR", "VTRS", "VZ", "WAB", "WAT", "WBA", "WBD", "WBS", "WCC", "WDAY",
    "WDC", "WEC", "WELL", "WEX", "WFC", "WHR", "WM", "WMB", "WMT", "WRB",
    "WRK", "WSC", "WSO", "WST", "WTW", "WWD", "WY", "WYNN", "XEL", "XOM",
    "XRAY", "XYL", "YUM", "ZBH", "ZBRA", "ZION", "ZTS"
]


st.set_page_config(
    page_title="Build Your Smart Investment Portfolio",
    page_icon="💹",
    layout="wide",
)

# Initialize session state for selected tickers
if "selected_tickers" not in st.session_state:
    # Default tickers
    st.session_state.selected_tickers = ["AAPL", "MSFT", "GOOG"]


def filter_tickers(search_term: str, ticker_list: List[str], max_results: int = 10) -> List[str]:
    """Filter tickers based on case-insensitive search term."""
    if not search_term:
        return ticker_list[:max_results]
    search_upper = search_term.upper().strip()
    filtered = [t for t in ticker_list if search_upper in t.upper()]
    return filtered[:max_results]


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_ticker_suggestions(search_term: str) -> List[str]:
    """
    Optional: Fetch ticker suggestions dynamically from a public API.
    Currently returns empty list - can be implemented with yfinance or other APIs.
    """
    # Example implementation (commented out):
    # try:
    #     # Using yfinance to search (if available)
    #     # Note: yfinance doesn't have a direct search API, so this is a placeholder
    #     # You could use other APIs like Alpha Vantage, IEX Cloud, etc.
    #     pass
    # except Exception:
    #     pass
    return []


def ticker_autocomplete_component() -> str:
    """Create an autocomplete ticker selector component."""
    st.write("**Search and select stock tickers**")
    
    # Search input
    search_term = st.text_input(
        "Type to search tickers (e.g., 'AA' for AAPL, AAL, etc.)",
        value="",
        key="ticker_search",
        placeholder="Start typing...",
        help="Type letters to see matching stock tickers. Click suggestions to add them, or type a ticker and press Enter to add manually.",
    )
    
    # Filter tickers based on search
    filtered_tickers = filter_tickers(search_term, POPULAR_TICKERS, max_results=15)
    
    # Show suggestions if there's a search term
    if search_term.strip():
        search_upper = search_term.strip().upper()
        # Check if the search term itself is a valid ticker (exact match or in filtered list)
        exact_match = search_upper in POPULAR_TICKERS
        
        if filtered_tickers:
            st.caption(f"Found {len(filtered_tickers)} matching ticker(s):")
            # Create columns for ticker buttons
            cols = st.columns(min(5, len(filtered_tickers)))
            for idx, ticker in enumerate(filtered_tickers):
                col_idx = idx % 5
                with cols[col_idx]:
                    if st.button(ticker, key=f"suggest_{ticker}", use_container_width=True):
                        if ticker not in st.session_state.selected_tickers:
                            st.session_state.selected_tickers.append(ticker)
                            st.rerun()
            
            # If search term is exact match but not in filtered (shouldn't happen), add it
            if exact_match and search_upper not in st.session_state.selected_tickers:
                if st.button(f"Add '{search_upper}'", key="add_exact_match", type="primary"):
                    st.session_state.selected_tickers.append(search_upper)
                    st.rerun()
        else:
            # No matches found, but allow manual entry if it looks like a ticker (3-5 uppercase letters)
            if len(search_upper) >= 1 and len(search_upper) <= 5 and search_upper.isalpha():
                st.caption(f"No matching tickers found. You can add '{search_upper}' manually:")
                if st.button(f"Add '{search_upper}' as ticker", key="add_manual_ticker", type="primary"):
                    if search_upper not in st.session_state.selected_tickers:
                        st.session_state.selected_tickers.append(search_upper)
                        st.rerun()
            else:
                st.caption("No matching tickers found. Try a different search or enter a valid ticker symbol (3-5 letters).")
    
    # Display selected tickers
    if st.session_state.selected_tickers:
        st.write(f"**Selected tickers ({len(st.session_state.selected_tickers)}):**")
        # Display tickers in a cleaner format
        ticker_display = " • ".join(st.session_state.selected_tickers)
        st.markdown(f"`{ticker_display}`")
        
        # Remove buttons in a grid
        num_selected = len(st.session_state.selected_tickers)
        remove_cols = st.columns(min(6, num_selected))
        for idx, ticker in enumerate(st.session_state.selected_tickers):
            col_idx = idx % 6
            with remove_cols[col_idx]:
                if st.button(f"Remove {ticker}", key=f"remove_{ticker}", use_container_width=True):
                    st.session_state.selected_tickers.remove(ticker)
                    st.rerun()
        
        # Clear all button
        if st.button("🗑️ Clear all selections", key="clear_all_tickers", type="secondary", use_container_width=True):
            st.session_state.selected_tickers = []
            st.rerun()
    else:
        st.info("💡 No tickers selected. Start typing above to search and add tickers.")
    
    # Return comma-separated string for backend compatibility
    return ", ".join(st.session_state.selected_tickers)


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

# ---- Step 1 ----
st.subheader("Step 1: Pick your stocks and time period")
st.caption("Search and select stock tickers using autocomplete, then choose the date range for historical data.")

col_ticker, col_dates = st.columns([1, 1])
with col_ticker:
    ticker_input = ticker_autocomplete_component()
today = date.today()
default_start = today - timedelta(days=365 * 5)
with col_dates:
    col_start, col_end = st.columns(2)
    start_date = col_start.date_input("Start date", value=default_start)
    end_date = col_end.date_input("End date", value=today)

# ---- Step 2 ----
st.subheader("Step 2: Set your preferences")
st.caption("These settings affect how we measure returns and risk.")

col_rf, col_freq, col_short = st.columns(3)
with col_rf:
    risk_free_rate_input = st.number_input(
        "Risk-free rate (% per year)",
        min_value=-5.0,
        max_value=20.0,
        value=2.0,
        step=0.1,
        help="Return from a safe investment (e.g. fixed deposit or government bond). Used to compare your portfolio.",
    ) / 100
    st.info("Think of this as the return you’d get from a safe option like an FD or government bond. We use it to see how much extra return you get for taking risk.")

frequency_map = {
    "Daily (252 trading days)": 252,
    "Weekly (52 weeks)": 52,
    "Monthly (12 months)": 12,
}
with col_freq:
    period_label = st.selectbox(
        "Return frequency",
        list(frequency_map.keys()),
        index=0,
        help="How often we treat returns: daily (most data), weekly, or monthly.",
    )
    periods_per_year = frequency_map[period_label]
    st.caption("Daily uses more data; monthly is smoother. Pick what matches how you think about returns.")

with col_short:
    allow_short_sales = st.toggle("Allow short selling", value=False)
    st.caption("Short selling means betting a stock will fall. Leave off for normal long-only investing.")

if not ticker_input.strip() or len(st.session_state.selected_tickers) == 0:
    st.info("Please select at least one ticker symbol to begin.")
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

# ---- Step 3 ----
st.subheader("Step 3: Your results")
st.caption("Below: a preview of the price data we used, then your portfolio options.")

with st.expander("Preview: historical price data (last 5 rows)", expanded=False):
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

# Minimum risk (min vol) and high return (max return) points on the frontier
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

frontier_plot = draw_frontier_plot(
    frontier_df, cml_data, risk_free_rate_input, min_vol_point, max_ret_point
)
st.pyplot(frontier_plot)

# ---- What This Means For You ----
st.subheader("What This Means For You")
st.caption("We recommend the **Balanced** portfolio (best risk–return mix) unless you prefer lower risk or higher return.")

example_amount = 100_000  # ₹1,00,000
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

col_summary1, col_summary2 = st.columns(2)
with col_summary1:
    st.metric("Expected return (annual)", f"{tan_ret:.1%}")
    st.caption("Average return you might expect per year based on history.")
    st.metric("Risk level", risk_label)
    st.caption(risk_explanation)
with col_summary2:
    st.write("**Suggested allocation (Balanced portfolio)**")
    for asset, w in zip(mean_returns.index, tangency_weights):
        st.write(f"- **{asset}**: {w:.0%}")
    st.write("")
    st.write(f"**If you invest ₹1,00,000:**")
    for asset, w in zip(mean_returns.index, tangency_weights):
        amt = example_amount * w
        st.write(f"- **{asset}**: ₹{amt:,.0f}")
st.divider()

# ---- Technical Details (expandable) ----
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
