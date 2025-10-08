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
    
    # Debug: Print the actual columns structure
    print(f"Data columns: {data.columns}")
    print(f"Data columns type: {type(data.columns)}")
    print(f"Data shape: {data.shape}")
    
    # Handle MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        print("MultiIndex columns detected")
        # If MultiIndex, get the first level (column) and second level (ticker)
        adj_close_col = None
        for col in data.columns:
            print(f"MultiIndex col: {col}")
            if col[0] == 'Adj Close':  # First element is column name
                adj_close_col = col
                break
        if adj_close_col is None:
            # Try Close if Adj Close not found
            for col in data.columns:
                if col[0] == 'Close':  # First element is column name
                    adj_close_col = col
                    break
        if adj_close_col is None:
            raise ValueError(f"Adjusted Close or Close not found in returned data. Available columns: {list(data.columns)}")
        adj = data[adj_close_col].copy()
    else:
        print("Regular columns detected")
        print(f"Available columns: {list(data.columns)}")
        # Handle regular columns - try multiple possible names
        adj = None
        possible_names = ['Adj Close', 'Close', 'adj_close', 'close', 'Adj_Close', 'Close_Price']
        
        for col_name in possible_names:
            if col_name in data.columns:
                adj = data[col_name].copy()
                print(f"Found column: {col_name}")
                break
        
        if adj is None:
            # If no standard names found, try to find any column that might be price data
            for col in data.columns:
                if 'close' in str(col).lower() or 'price' in str(col).lower():
                    adj = data[col].copy()
                    print(f"Found price-like column: {col}")
                    break
        
        if adj is None:
            raise ValueError(f"No price column found. Available columns: {list(data.columns)}")
    
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


def compute_forward_returns(adj: pd.Series, horizon: int = 1) -> pd.Series:
    """Compute n-day forward simple returns."""
    fwd = adj.pct_change(periods=horizon).shift(-horizon)
    fwd.name = f"fwd{horizon}d_ret"
    return fwd


def percentile_series(series: pd.Series) -> pd.Series:
    """Return percentile rank of each value within the series (0–100)."""
    s = series.dropna()
    ranks = s.rank(pct=True) * 100
    return ranks.reindex(series.index)


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
    p100 = np.percentile(s.values, 100)
    ax.axvspan(p95, p100, alpha=0.1, color='red', label='Panic Zone (95th-100th)')

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
    
    forward_horizon = st.number_input("Forward Return Horizon (days)", 
                                     min_value=1, 
                                     max_value=252, 
                                     value=1, 
                                     step=1,
                                     help="Number of days ahead to analyze forward returns (1-252 days)")
    
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
            fwd_returns = compute_forward_returns(adj, horizon=forward_horizon)
            
            # Fetch VIX data for the same period
            try:
                vix_data = yf.download("^VIX", start=start_date.strftime('%Y-%m-%d'), 
                                     end=end_date.strftime('%Y-%m-%d'), progress=False)
                if not vix_data.empty and 'Close' in vix_data.columns:
                    vix_close = vix_data['Close'].copy()
                    vix_close.name = 'vix_close'
                else:
                    vix_close = None
            except:
                vix_close = None
            
            P_today = float(adj.iloc[-1])
            
            # Always scale to today's price
            usd_vol_last = vol_ret * P_today
            usd_vol_last.name = f'usd_vol{window}_scaled_to_today_last'
            usd_vol_mean = vol_ret * P_today
            usd_vol_mean.name = f'usd_vol{window}_scaled_to_today_mean'
            
            # Compute volatility percentile ranks and forward return analysis
            vol_pct_rank = percentile_series(vol_ret)
            
            # Custom percentile breakpoints with 1% granularity and fine extreme tail
            vol_bins = list(range(0, 95, 1)) + [95, 99, 99.5, 99.9, 100]
            vol_labels = [f"{i}–{i+1}" for i in range(0, 95, 1)] + [
                "95–99", "99–99.5", "99.5–99.9", "99.9–100"
            ]
            vol_cat = pd.cut(vol_pct_rank, bins=vol_bins, labels=vol_labels, include_lowest=True)
            
            # Combine volatility percentile and forward return
            vol_forward_df = pd.DataFrame({
                "vol": vol_ret,
                "vol_pct": vol_pct_rank,
                "vol_bin": vol_cat,
                f"fwd{forward_horizon}d": fwd_returns
            })
            
            # Add VIX data if available
            if vix_close is not None:
                vol_forward_df["vix"] = vix_close.reindex(vol_forward_df.index)
            
            vol_forward_df = vol_forward_df.dropna(subset=["vol", f"fwd{forward_horizon}d"])
            
            # Aggregate stats
            vol_forward_summary = (
                vol_forward_df
                .groupby("vol_bin")[f"fwd{forward_horizon}d"]
                .agg(["mean", "median", "count"])
                .reset_index()
            )
            
            # Validation checksum for observation counts
            N = len(adj)
            w = window
            h = forward_horizon
            expected_valid_obs = max(0, N - w - h)
            
            # Rows where BOTH vol and forward return are available
            aligned_rows = vol_forward_df.dropna(subset=["vol", f"fwd{forward_horizon}d"]).shape[0]
            
            # Sum of bin counts
            binned_count = int(vol_forward_summary["count"].sum()) if not vol_forward_summary.empty else 0
            
        
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
                
                # Forward Returns Analysis
                st.subheader(f"📈 {forward_horizon}-Day Forward Returns by Volatility Percentile")
                
                if not vol_forward_summary.empty:
                    fig3, ax = plt.subplots(figsize=(14, 8))
                    
                    # Create numeric x-axis for better plotting
                    vol_forward_summary['bin_numeric'] = vol_forward_summary['vol_bin'].apply(
                        lambda x: float(x.split('–')[0]) if '–' in str(x) else 0
                    ).astype(float)
                    vol_forward_summary_sorted = vol_forward_summary.sort_values('bin_numeric')
                    
                    # Plot mean and median as lines for better visibility with many bins
                    ax.plot(vol_forward_summary_sorted["bin_numeric"], 
                           vol_forward_summary_sorted["mean"] * 100, 
                           color="steelblue", linewidth=2, marker="o", markersize=3, 
                           label="Mean", alpha=0.8)
                    ax.plot(vol_forward_summary_sorted["bin_numeric"], 
                           vol_forward_summary_sorted["median"] * 100, 
                           color="crimson", linewidth=2, marker="s", markersize=3, 
                           label="Median", alpha=0.8)
                    
                    # Highlight extreme tail values with larger markers and annotations
                    extreme_bins = vol_forward_summary_sorted[vol_forward_summary_sorted["vol_bin"].isin(["95–99", "99–99.5", "99.5–99.9", "99.9–100"])]
                    if not extreme_bins.empty:
                        # Plot extreme values with larger, more prominent markers
                        ax.scatter(extreme_bins["bin_numeric"], 
                                 extreme_bins["mean"] * 100, 
                                 color="red", s=100, marker="D", 
                                 label="Extreme Tail (Mean)", alpha=0.9, zorder=5)
                        ax.scatter(extreme_bins["bin_numeric"], 
                                 extreme_bins["median"] * 100, 
                                 color="darkred", s=100, marker="^", 
                                 label="Extreme Tail (Median)", alpha=0.9, zorder=5)
                        
                        # Add annotations for extreme values
                        for _, row in extreme_bins.iterrows():
                            # Annotate with both percentile and return value
                            annotation_text = f"{row['vol_bin']}\n{row['mean']*100:.2f}%"
                            ax.annotate(annotation_text, 
                                      xy=(row["bin_numeric"], row["mean"] * 100),
                                      xytext=(5, 15), textcoords='offset points',
                                      fontsize=9, fontweight='bold', color='red',
                                      ha='left', va='bottom',
                                      bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='red', linewidth=1.5))
                    
                    ax.axhline(0, color="black", linewidth=1, alpha=0.7)
                    ax.set_xlabel("Volatility Percentile", fontsize=12, fontweight='bold')
                    ax.set_ylabel(f"{forward_horizon}-Day Forward Return (%)", fontsize=12, fontweight='bold')
                    ax.set_title(f"{ticker} {forward_horizon}-Day Forward Returns vs Realized Volatility Percentile", 
                               fontsize=14, fontweight='bold', pad=20)
                    ax.legend(fontsize=11)
                    ax.grid(True, alpha=0.3)
                    
                    # Set x-axis ticks for better readability
                    ax.set_xticks(range(0, 101, 10))
                    ax.set_xlim(0, 100)
                    
                    plt.tight_layout()
                    st.pyplot(fig3)
                    
                    # Win Rate vs Volatility Percentile Chart
                    st.subheader("🎯 Win Rate by Volatility Percentile")
                    
                    fig4, ax2 = plt.subplots(figsize=(14, 6))
                    
                    # Calculate win rates for all bins
                    win_rates = []
                    bin_centers = []
                    bin_labels = []
                    
                    for _, row in vol_forward_summary_sorted.iterrows():
                        bin_data = vol_forward_df[vol_forward_df["vol_bin"] == row["vol_bin"]]
                        if not bin_data.empty:
                            win_rate = (bin_data[f"fwd{forward_horizon}d"] > 0).mean() * 100
                            win_rates.append(win_rate)
                            bin_centers.append(row["bin_numeric"])
                            bin_labels.append(row["vol_bin"])
                    
                    # Plot win rate as a line with markers
                    ax2.plot(bin_centers, win_rates, color="green", linewidth=2, marker="o", 
                            markersize=4, label="Win Rate", alpha=0.8)
                    
                    # Add horizontal line at 50% (random chance)
                    ax2.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="50% (Random)")
                    
                    # Highlight extreme tail values
                    extreme_indices = [i for i, label in enumerate(bin_labels) if label in ["95–99", "99–99.5", "99.5–99.9", "99.9–100"]]
                    if extreme_indices:
                        extreme_centers = [bin_centers[i] for i in extreme_indices]
                        extreme_win_rates = [win_rates[i] for i in extreme_indices]
                        extreme_labels = [bin_labels[i] for i in extreme_indices]
                        
                        # Plot extreme values with larger markers
                        ax2.scatter(extreme_centers, extreme_win_rates, 
                                  color="red", s=100, marker="D", 
                                  label="Extreme Tail", alpha=0.9, zorder=5)
                        
                        # Add annotations for extreme values
                        for center, win_rate, label in zip(extreme_centers, extreme_win_rates, extreme_labels):
                            ax2.annotate(f"{label}\n{win_rate:.1f}%", 
                                       xy=(center, win_rate),
                                       xytext=(5, 10), textcoords='offset points',
                                       fontsize=9, fontweight='bold', color='red',
                                       ha='left', va='bottom',
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='red'))
                    
                    ax2.set_xlabel("Volatility Percentile", fontsize=12, fontweight='bold')
                    ax2.set_ylabel("Win Rate (%)", fontsize=12, fontweight='bold')
                    ax2.set_title(f"{ticker} Win Rate vs Realized Volatility Percentile ({forward_horizon}-Day Forward Returns)", 
                                fontsize=14, fontweight='bold', pad=20)
                    ax2.set_ylim(0, 100)
                    ax2.set_xticks(range(0, 101, 10))
                    ax2.set_xlim(0, 100)
                    ax2.grid(True, alpha=0.3)
                    ax2.legend(fontsize=11)
                    
                    plt.tight_layout()
                    st.pyplot(fig4)
                    
                    # VIX vs Volatility Percentile Chart
                    if vix_close is not None and 'vix' in vol_forward_df.columns:
                        st.subheader("📊 VIX vs Volatility Percentile")
                        
                        fig5, ax3 = plt.subplots(figsize=(14, 6))
                        
                        # Calculate VIX stats for all bins
                        vix_means = []
                        vix_medians = []
                        bin_centers_vix = []
                        bin_labels_vix = []
                        
                        for _, row in vol_forward_summary_sorted.iterrows():
                            bin_data = vol_forward_df[vol_forward_df["vol_bin"] == row["vol_bin"]]
                            if not bin_data.empty and 'vix' in bin_data.columns:
                                vix_bin_data = bin_data['vix'].dropna()
                                if not vix_bin_data.empty:
                                    vix_means.append(vix_bin_data.mean())
                                    vix_medians.append(vix_bin_data.median())
                                    bin_centers_vix.append(row["bin_numeric"])
                                    bin_labels_vix.append(row["vol_bin"])
                        
                        if vix_means:  # Only plot if we have VIX data
                            # Plot VIX mean and median
                            ax3.plot(bin_centers_vix, vix_means, color="purple", linewidth=2, marker="o", 
                                    markersize=4, label="VIX Mean", alpha=0.8)
                            ax3.plot(bin_centers_vix, vix_medians, color="orange", linewidth=2, marker="s", 
                                    markersize=4, label="VIX Median", alpha=0.8)
                            
                            # Highlight extreme tail values
                            extreme_indices_vix = [i for i, label in enumerate(bin_labels_vix) if label in ["95–99", "99–99.5", "99.5–99.9", "99.9–100"]]
                            if extreme_indices_vix:
                                extreme_centers_vix = [bin_centers_vix[i] for i in extreme_indices_vix]
                                extreme_means_vix = [vix_means[i] for i in extreme_indices_vix]
                                extreme_medians_vix = [vix_medians[i] for i in extreme_indices_vix]
                                extreme_labels_vix = [bin_labels_vix[i] for i in extreme_indices_vix]
                                
                                # Plot extreme values with larger markers
                                ax3.scatter(extreme_centers_vix, extreme_means_vix, 
                                          color="red", s=100, marker="D", 
                                          label="Extreme Tail (Mean)", alpha=0.9, zorder=5)
                                ax3.scatter(extreme_centers_vix, extreme_medians_vix, 
                                          color="darkred", s=100, marker="^", 
                                          label="Extreme Tail (Median)", alpha=0.9, zorder=5)
                                
                                # Add annotations for extreme values
                                for center, vix_mean, vix_median, label in zip(extreme_centers_vix, extreme_means_vix, extreme_medians_vix, extreme_labels_vix):
                                    ax3.annotate(f"{label}\nMean: {vix_mean:.1f}\nMed: {vix_median:.1f}", 
                                               xy=(center, vix_mean),
                                               xytext=(5, 15), textcoords='offset points',
                                               fontsize=8, fontweight='bold', color='red',
                                               ha='left', va='bottom',
                                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='red'))
                            
                            ax3.set_xlabel("Volatility Percentile", fontsize=12, fontweight='bold')
                            ax3.set_ylabel("VIX Level", fontsize=12, fontweight='bold')
                            ax3.set_title(f"VIX vs {ticker} Volatility Percentile", 
                                        fontsize=14, fontweight='bold', pad=20)
                            ax3.set_xticks(range(0, 101, 10))
                            ax3.set_xlim(0, 100)
                            ax3.grid(True, alpha=0.3)
                            ax3.legend(fontsize=11)
                            
                            plt.tight_layout()
                            st.pyplot(fig5)
                        else:
                            st.warning("⚠️ Insufficient VIX data for analysis")
                    else:
                        st.info("ℹ️ VIX data not available for the selected period")
                    
                    # Interpretation metrics for extreme volatility bins
                    st.markdown("### 🧭 Extreme-Tail Behavior")
                    extreme_bins = vol_forward_summary[vol_forward_summary["vol_bin"].isin(["95–99", "99–99.5", "99.5–99.9", "99.9–100"])]
                    
                    if not extreme_bins.empty:
                        for _, row in extreme_bins.iterrows():
                            st.markdown(
                                f"**{row['vol_bin']}% vol bin** — "
                                f"Mean = {row['mean']*100:.3f}% | Median = {row['median']*100:.3f}% | n = {int(row['count'])}"
                            )
                    
                    # Summary insights
                    if len(vol_forward_summary) >= 2:
                        high_vol_bins = vol_forward_summary[vol_forward_summary["vol_bin"].isin(["95–99", "99–99.5", "99.5–99.9", "99.9–100"])]
                        low_vol_bins = vol_forward_summary[vol_forward_summary["bin_numeric"].astype(float) <= 50]
                        
                        if not high_vol_bins.empty and not low_vol_bins.empty:
                            high_vol_mean = high_vol_bins["mean"].mean() * 100
                            low_vol_mean = low_vol_bins["mean"].mean() * 100
                            
                            st.markdown("### 📊 Key Insights")
                            st.markdown(f"📈 **High Volatility Regime** (95%+): Average {forward_horizon}-day return = **{high_vol_mean:.3f}%**")
                            st.markdown(f"📉 **Low Volatility Regime** (0-50%): Average {forward_horizon}-day return = **{low_vol_mean:.3f}%**")
                            
                            if high_vol_mean > low_vol_mean:
                                st.markdown(f"🔄 **Mean Reversion Signal**: High volatility periods tend to be followed by positive {forward_horizon}-day returns")
                            else:
                                st.markdown(f"📉 **Momentum Signal**: High volatility periods tend to be followed by negative {forward_horizon}-day returns")
                    
                    # Summary table for key percentile ranges including extreme tail
                    st.markdown("### 📋 Summary by Percentile Range")
                    summary_ranges = []
                    
                    # Standard 10% ranges (0-90%)
                    for start in range(0, 90, 10):
                        end = start + 10
                        range_bins = vol_forward_summary[
                            (vol_forward_summary["bin_numeric"].astype(float) >= start) & 
                            (vol_forward_summary["bin_numeric"].astype(float) < end)
                        ]
                        if not range_bins.empty:
                            # Calculate win rate for this range
                            range_data = vol_forward_df[
                                (vol_forward_df["vol_pct"].astype(float) >= start) & 
                                (vol_forward_df["vol_pct"].astype(float) < end)
                            ]
                            win_rate = (range_data[f"fwd{forward_horizon}d"] > 0).mean() * 100 if not range_data.empty else 0
                            
                            summary_ranges.append({
                                "Percentile Range": f"{start}-{end}%",
                                "Mean Return (%)": f"{range_bins['mean'].mean()*100:.3f}",
                                "Median Return (%)": f"{range_bins['median'].mean()*100:.3f}",
                                "Observations": int(range_bins['count'].sum()),
                                "Win Rate (%)": f"{win_rate:.1f}"
                            })
                    
                    # 90-95% range
                    range_bins = vol_forward_summary[
                        (vol_forward_summary["bin_numeric"].astype(float) >= 90) & 
                        (vol_forward_summary["bin_numeric"].astype(float) < 95)
                    ]
                    if not range_bins.empty:
                        # Calculate win rate for 90-95% range
                        range_data = vol_forward_df[
                            (vol_forward_df["vol_pct"].astype(float) >= 90) & 
                            (vol_forward_df["vol_pct"].astype(float) < 95)
                        ]
                        win_rate = (range_data[f"fwd{forward_horizon}d"] > 0).mean() * 100 if not range_data.empty else 0
                        
                        summary_ranges.append({
                            "Percentile Range": "90-95%",
                            "Mean Return (%)": f"{range_bins['mean'].mean()*100:.3f}",
                            "Median Return (%)": f"{range_bins['median'].mean()*100:.3f}",
                            "Observations": int(range_bins['count'].sum()),
                            "Win Rate (%)": f"{win_rate:.1f}"
                        })
                    
                    # Individual extreme tail bins (95-99%, 99-99.5%, 99.5-99.9%, 99.9-100%)
                    extreme_bin_names = ["95–99", "99–99.5", "99.5–99.9", "99.9–100"]
                    for bin_name in extreme_bin_names:
                        bin_data = vol_forward_summary[vol_forward_summary["vol_bin"] == bin_name]
                        if not bin_data.empty:
                            # Calculate win rate for this extreme bin
                            extreme_data = vol_forward_df[vol_forward_df["vol_bin"] == bin_name]
                            win_rate = (extreme_data[f"fwd{forward_horizon}d"] > 0).mean() * 100 if not extreme_data.empty else 0
                            
                            summary_ranges.append({
                                "Percentile Range": f"{bin_name}%",
                                "Mean Return (%)": f"{bin_data['mean'].iloc[0]*100:.3f}",
                                "Median Return (%)": f"{bin_data['median'].iloc[0]*100:.3f}",
                                "Observations": int(bin_data['count'].iloc[0]),
                                "Win Rate (%)": f"{win_rate:.1f}"
                            })
                    
                    if summary_ranges:
                        summary_df = pd.DataFrame(summary_ranges)
                        st.dataframe(summary_df, use_container_width=True)
                    
                else:
                    st.warning("⚠️ Insufficient data for forward returns analysis")
        
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
            
            # Prepare output dataframe with unique names
            out_data = {}
            out_data['adj_close'] = adj.copy()
            out_data[f'ret_{returns_type}'] = ret.copy()
            out_data[f'vol{window}_ret'] = vol_ret.copy()
            out_data[f'usd_vol{window}_last'] = usd_vol_last.copy()
            out_data[f'usd_vol{window}_mean'] = usd_vol_mean.copy()
            out_data[f'fwd{forward_horizon}d_ret'] = fwd_returns.copy()
            out_data['vol_pct_rank'] = vol_pct_rank.copy()
            out_data['vol_bin'] = vol_cat.copy()
            
            out = pd.DataFrame(out_data)
            
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
                    "forward_horizon": forward_horizon,
                    "scale_to_today": True
                },
                "usd_vol_last": stats_last,
                "usd_vol_mean": stats_mean,
                "forward_returns_by_vol_bin": vol_forward_summary.to_dict('records') if not vol_forward_summary.empty else []
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
        This app computes the empirical distribution of rolling standard deviation in USD and analyzes forward returns:
        
        1. **Fetch Data**: Downloads adjusted close prices from Yahoo Finance
        2. **Calculate Returns**: Computes daily log or simple returns
        3. **Rolling Volatility**: Calculates rolling standard deviation of returns
        4. **USD Conversion**: Converts volatility to USD by scaling all historical volatilities to today's price level
        5. **Forward Returns Analysis**: Analyzes configurable forward returns (1-20 days) grouped by volatility percentile bins
        
        The histograms show where current volatility ranks historically, with percentile markers
        and a "panic zone" highlighting extreme volatility periods (95th-99th percentile).
        
        The forward returns analysis reveals whether high volatility periods tend to be followed by
        mean reversion (positive returns) or momentum (negative returns) over the selected time horizon.
        """)
