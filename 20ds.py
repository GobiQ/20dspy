#!/usr/bin/env python3
"""
Streamlit app for computing and visualizing 20-day USD volatility distributions.
Run with: streamlit run app.py
"""

import streamlit as st
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import io

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    st.warning("⚠️ Matplotlib not installed. Visualizations will be disabled. Please add 'matplotlib' to requirements.txt")

# Page config
st.set_page_config(page_title="USD Volatility Analyzer", page_icon="📊", layout="wide")

# Helper functions from original script
def _percentile_rank(values: np.ndarray, x: float) -> float:
    """Return the percentile rank of x within values (0-100)."""
    v = np.asarray(values)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float('nan')
    v_sorted = np.sort(v)
    k = np.searchsorted(v_sorted, x, side="right")
    return 100.0 * k / v_sorted.size


@st.cache_data
def fetch_adj_close(ticker: str, start: str = None, end: str = None) -> pd.Series:
    """Fetch Adjusted Close with yfinance."""
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if data.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker or date range.")
    if 'Adj Close' not in data.columns:
        if 'Close' in data.columns:
            adj = data['Close'].copy()
            adj.name = 'adj_close'
            return adj
        raise ValueError("Adjusted Close not found in returned data.")
    adj = data['Adj Close'].copy()
    adj.name = 'adj_close'
    return adj


def compute_returns(adj: pd.Series, kind: str = 'log') -> pd.Series:
    """Compute daily returns from Adjusted Close."""
    if kind == 'log':
        ret = np.log(adj).diff()
    else:
        ret = adj.pct_change()
    ret.name = f'ret_{kind}'
    return ret


def rolling_volatility(ret: pd.Series, window: int = 20) -> pd.Series:
    """Rolling sample std of daily returns."""
    vol = ret.rolling(window=window, min_periods=window).std(ddof=1)
    vol.name = f'vol{window}_ret'
    return vol


def rolling_mean_price(adj: pd.Series, window: int = 20) -> pd.Series:
    """Compute the mean price over the rolling window."""
    mean_price = adj.rolling(window=window, min_periods=window).mean()
    mean_price.name = 'price_mean'
    return mean_price


def summarize_distribution(series: pd.Series) -> dict:
    """Return summary stats and percentiles."""
    s = series.dropna()
    if s.empty:
        return {}
    percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    q = np.percentile(s.values, percentiles)
    stat = {
        "count": int(s.size),
        "mean": float(np.mean(s.values)),
        "std": float(np.std(s.values, ddof=1)) if s.size > 1 else float('nan'),
        "min": float(np.min(s.values)),
        "max": float(np.max(s.values)),
        "percentiles": {str(p): float(v) for p, v in zip(percentiles, q)},
    }
    return stat


def create_histogram(series: pd.Series, title: str, marker_value: float = None):
    """Create histogram plot."""
    if not MATPLOTLIB_AVAILABLE:
        return None
        
    s = series.dropna()
    if s.empty:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Histogram
    n, bins, patches = ax.hist(s.values, bins=50, alpha=0.7, color='steelblue',
                               edgecolor='black', linewidth=0.5)

    # Stats
    mean_val = float(s.mean())
    median_val = float(s.median())
    std_val = float(s.std())
    p10, p25, p75 = np.percentile(s.values, [10, 25, 75])
    p90, p95, p99 = np.percentile(s.values, [90, 95, 99])
    p99_5, p99_9 = np.percentile(s.values, [99.5, 99.9])

    stats_text = (
        f"Mean: ${mean_val:.2f}\nStd: ${std_val:.2f}\nMedian: ${median_val:.2f}\n\n"
        f"P10: ${p10:.2f}\nP25: ${p25:.2f}\nP75: ${p75:.2f}\nP90: ${p90:.2f}\n"
        f"P95: ${p95:.2f}\nP99: ${p99:.2f}\nP99.5: ${p99_5:.2f}\nP99.9: ${p99_9:.2f}\n\nCount: {len(s):,}"
    )
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            va='top', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Percentile lines
    percentiles = np.concatenate([np.arange(10, 100, 10), [95, 99, 99.5, 99.9]])
    percentile_values = np.percentile(s.values, percentiles)
    colors = plt.cm.viridis(np.linspace(0, 1, len(percentiles)))
    
    for i, (p, val) in enumerate(zip(percentiles, percentile_values)):
        ax.axvline(val, color=colors[i], linestyle='-', linewidth=1.0, alpha=0.6)
        ax.text(val, ax.get_ylim()[1] * 0.95, f'{p}th',
                rotation=90, ha='center', va='top', fontsize=8,
                color=colors[i], fontweight='bold')

    # Panic zone
    ax.axvspan(p95, p99, alpha=0.1, color='red', label='Panic Zone (95th-99th)')

    # Mean & median
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2.5, alpha=0.9, label=f"Mean ${mean_val:.2f}")
    ax.axvline(median_val, color='orange', linestyle='--', linewidth=2.5, alpha=0.9, label=f"Median ${median_val:.2f}")

    # Today marker
    if marker_value is not None and np.isfinite(marker_value):
        pr = _percentile_rank(s.values, marker_value)
        ax.axvline(marker_value, color='crimson', linewidth=4.0, alpha=0.95)
        marker_text = f"Today\n${marker_value:.2f}\n({pr:.1f}th pct)"
        ax.text(marker_value, ax.get_ylim()[1] * 0.85,
                marker_text,
                rotation=90, ha='center', va='top', fontsize=11,
                color='crimson', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='crimson'))

    ax.set_xlabel('USD Volatility', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.2f}'))
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(loc='upper left', ncol=1, fontsize=9)

    plt.tight_layout()
    return fig


# Streamlit App
st.title("📊 USD Volatility Distribution Analyzer")
st.markdown("Analyze the empirical distribution of rolling standard deviation in USD for any ticker")

# Sidebar for inputs
with st.sidebar:
    st.header("⚙️ Configuration")
    
    ticker = st.text_input("Ticker Symbol", value="SPY", help="Enter a valid ticker symbol")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(1993, 1, 29))
    with col2:
        end_date = st.date_input("End Date", value=datetime.today())
    
    window = st.slider("Rolling Window (days)", min_value=5, max_value=60, value=20)
    
    returns_type = st.selectbox("Return Type", options=["log", "simple"], index=0)
    
    scale_to_today = st.checkbox("Scale to Today's Price", value=False, 
                                 help="Scale all historical volatilities to today's price level")
    
    run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

# Main content
if run_analysis:
    try:
        with st.spinner(f"Fetching data for {ticker}..."):
            # Fetch data
            adj = fetch_adj_close(ticker, start=start_date.strftime('%Y-%m-%d'), 
                                end=end_date.strftime('%Y-%m-%d'))
            
            # Compute metrics
            ret = compute_returns(adj, kind=returns_type)
            vol_ret = rolling_volatility(ret, window=window)
            
            P_today = float(adj.iloc[-1])
            
            if scale_to_today:
                usd_vol_last = vol_ret * P_today
                usd_vol_last.name = f'usd_vol{window}_scaled_to_today_last'
                usd_vol_mean = vol_ret * P_today
                usd_vol_mean.name = f'usd_vol{window}_scaled_to_today_mean'
            else:
                usd_vol_last = vol_ret * adj
                usd_vol_last.name = f'usd_vol{window}_last'
                price_mean = rolling_mean_price(adj, window=window)
                usd_vol_mean = vol_ret * price_mean
                usd_vol_mean.name = f'usd_vol{window}_mean'
        
        # Display key metrics
        st.success(f"✅ Successfully analyzed {ticker}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${P_today:.2f}")
        with col2:
            today_vol = float(vol_ret.dropna().iloc[-1]) if not vol_ret.dropna().empty else 0
            st.metric("Current Vol (Return)", f"{today_vol:.4f}")
        with col3:
            today_usd = float(usd_vol_last.dropna().iloc[-1]) if not usd_vol_last.dropna().empty else 0
            st.metric("Current USD Vol", f"${today_usd:.2f}")
        with col4:
            pct_rank = _percentile_rank(usd_vol_last.dropna().values, today_usd)
            st.metric("Percentile Rank", f"{pct_rank:.1f}th")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Visualizations", "📊 Statistics", "📋 Data", "💾 Downloads"])
        
        with tab1:
            st.subheader("USD Volatility Distributions")
            
            if not MATPLOTLIB_AVAILABLE:
                st.error("📊 Visualizations require matplotlib. Please create a requirements.txt file with: matplotlib, yfinance, pandas, numpy")
            else:
                # Calculate today's markers
                vr = vol_ret.dropna()
                today_vol_ret = vr.iloc[-1] if not vr.empty else float('nan')
                today_usd_last = float(today_vol_ret * P_today) if np.isfinite(float(today_vol_ret)) else None
                
                if not scale_to_today and 'price_mean' in locals():
                    last_price_mean = price_mean.iloc[-1] if not price_mean.dropna().empty else float('nan')
                    today_usd_mean = float(today_vol_ret * last_price_mean) if np.isfinite(float(today_vol_ret)) and np.isfinite(float(last_price_mean)) else None
                else:
                    today_usd_mean = today_usd_last
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig1 = create_histogram(usd_vol_last, 
                                          f"{ticker} {window}-day USD Volatility (Last Price)",
                                          marker_value=today_usd_last)
                    if fig1:
                        st.pyplot(fig1)
                
                with col2:
                    fig2 = create_histogram(usd_vol_mean,
                                          f"{ticker} {window}-day USD Volatility (Mean Price)",
                                          marker_value=today_usd_mean)
                    if fig2:
                        st.pyplot(fig2)
        
        with tab2:
            st.subheader("Statistical Summary")
            
            stats_last = summarize_distribution(usd_vol_last)
            stats_mean = summarize_distribution(usd_vol_mean)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Last Price Scaling**")
                if stats_last:
                    st.json(stats_last)
            
            with col2:
                st.markdown("**Mean Price Scaling**")
                if stats_mean:
                    st.json(stats_mean)
        
        with tab3:
            st.subheader("Time Series Data")
            
            # Prepare output dataframe
            adj_copy = adj.copy()
            
            # Ensure all Series have unique names
            ret.name = f'ret_{returns_type}'
            vol_ret.name = f'vol{window}_ret'
            
            if not scale_to_today and 'price_mean' in locals():
                out = pd.concat([adj_copy, ret, vol_ret, usd_vol_last, usd_vol_mean, price_mean], axis=1)
            else:
                out = pd.concat([adj_copy, ret, vol_ret, usd_vol_last, usd_vol_mean], axis=1)
            
            st.dataframe(out.tail(50), use_container_width=True)
            st.caption(f"Showing last 50 rows of {len(out)} total")
        
        with tab4:
            st.subheader("Download Results")
            
            # CSV download
            csv_buffer = io.StringIO()
            out.to_csv(csv_buffer, index_label='date')
            st.download_button(
                label="📥 Download CSV",
                data=csv_buffer.getvalue(),
                file_name=f"{ticker}_{window}d_usd_vol.csv",
                mime="text/csv"
            )
            
            # JSON stats download
            stats_dict = {
                "meta": {
                    "ticker": ticker,
                    "start": start_date.strftime('%Y-%m-%d'),
                    "end": end_date.strftime('%Y-%m-%d'),
                    "window": window,
                    "returns": returns_type,
                    "scale_to_today": scale_to_today
                },
                "usd_vol_last": stats_last,
                "usd_vol_mean": stats_mean
            }
            
            st.download_button(
                label="📥 Download Statistics (JSON)",
                data=json.dumps(stats_dict, indent=2),
                file_name=f"{ticker}_{window}d_usd_vol_stats.json",
                mime="application/json"
            )
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

else:
    st.info("👈 Configure your analysis parameters in the sidebar and click 'Run Analysis'")
    
    # Show example
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        This app computes the empirical distribution of rolling standard deviation in USD:
        
        1. **Fetch Data**: Downloads adjusted close prices from Yahoo Finance
        2. **Calculate Returns**: Computes daily log or simple returns
        3. **Rolling Volatility**: Calculates rolling standard deviation of returns
        4. **USD Conversion**: Converts volatility to USD using:
           - **Last Price**: Multiply by the last price in each window
           - **Mean Price**: Multiply by the mean price over each window
        
        The histograms show where current volatility ranks historically, with percentile markers
        and a "panic zone" highlighting extreme volatility periods (95th-99th percentile).
        """)
