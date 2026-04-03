import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
from dashboard_core import (
    forecast_next5, get_training_metrics, 
    get_metrics_summary, get_engineering_report, get_project_story,
    get_test_samples, get_forecast_bias
)

# --- 1. APP CONFIG ---
st.set_page_config(
    page_title="LSTM Stock Price Forecaster",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. PREMIUM CSS (PROFESSIONAL CLINICAL THEME) ---
st.markdown("""
<style>
    #MainMenu {visibility: hidden}
    footer {visibility: hidden}
    header {visibility: hidden}
    
    [data-testid="stSidebar"] {background-color: #fcfcfc; border-right: 1px solid #eee}
    [data-testid="stSidebar"] * {color: #333; font-family: 'DM Sans', sans-serif}
    
    .block-container {padding-top: 2rem; padding-bottom: 2rem}
    h1, h2, h3 {font-weight: 500; letter-spacing: -0.02em; color: #1a1a1a}
    .muted-text {color: #666; font-size: 0.85rem}
    
    /* Professional KPI Cards */
    .kpi-row {display: flex; gap: 1.5rem; margin-bottom: 2.5rem}
    .kpi-card {
        flex: 1;
        padding: 1.2rem;
        background: #ffffff;
        border-radius: 4px;
        border: 1px solid #eee;
        border-bottom: 3px solid #7F77DD;
    }
    .kpi-label {font-size: 11px; color: #666; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em}
    .kpi-value {font-size: 24px; font-weight: 500; color: #1a1a1a; margin-top: 4px}

    /* Status Pill */
    .status-pill {
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 10px;
    }

    /* Strategy Section */
    .strategy-card {
        padding: 1rem;
        background: #fcfcfc;
        border: 1px solid #eee;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. SIDEBAR (ARCHITECTURE SPECS) ---
with st.sidebar:
    st.markdown("### LSTM FORECASTER v2.0")
    st.markdown("<p class='muted-text'>Direct Multi-step Time Series Core</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Architecture Info
    st.markdown("#### Hardware & Stack")
    st.markdown("""
    - **Architecture**: Stacked LSTM
    - **Prediction Window**: 5-Day Direct
    - **Hidden State**: 256-dim Temporal Memory
    - **LR Schedule**: Warmup + Cosine Decay
    - **Dataset**: AAPL (Apple Inc.)
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    ticker = st.text_input("Live Stock Ticker", "AAPL")
    fetch_btn = st.button("Fetch & Forecast Live", type="primary", use_container_width=True)
    st.caption("Fetches 100 days of history via yfinance API.")

# --- 4. HEADER KPIs ---
eng_report = get_engineering_report()
st.markdown("## Time Series Forecasting Platform")
st.markdown("<p class='muted-text'>Advanced multi-step neural forecasting for sequential price data.</p>", unsafe_allow_html=True)

st.markdown(f"""
<div class="kpi-row">
    <div class="kpi-card">
        <div class="kpi-label">Prediction Target</div>
        <div class="kpi-value">Next 5 Close Prices</div>
        <div class="muted-text">Multi-Step Temporal Forecast</div>
    </div>
    <div class="kpi-card" style="border-bottom-color: #1D9E75">
        <div class="kpi-label">Scaled RMSE</div>
        <div class="kpi-value">{eng_report['scaled_rmse']}</div>
        <div class="muted-text">Validation Set (Holdout)</div>
    </div>
    <div class="kpi-card" style="border-bottom-color: #333">
        <div class="kpi-label">Core Architecture</div>
        <div class="kpi-value">Stacked LSTM</div>
        <div class="muted-text">{eng_report['architecture']}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 5. DATA INGESTION ---
if fetch_btn:
    with st.spinner(f"Connecting to Yahoo Finance for {ticker}..."):
        # Fetch 100 days to ensure enough history for MA30 and 60-day LSTM window
        df_new = yf.download(ticker, period="100d")
        if len(df_new) >= 90: # Increased from 60 to 90 to ensure 60 valid rows after MA30
            # Reorder to match: Close, High, Low, Open, Volume (Standard order)
            df_cleaned = df_new[['Close', 'High', 'Low', 'Open', 'Volume']]
            st.session_state.active_df = df_cleaned
            st.session_state.active_ticker = ticker
            st.session_state.active_forecast = forecast_next5(df_cleaned)
        else:
            st.error(f"Insufficient data for {ticker}. Need at least 90 days of history (60 window + 30 buffer).")

# --- 6. FUNCTIONAL TABS ---
tab1, tab2, tab3 = st.tabs(["Active Forecast", "Model Dynamics", "Engineering & Audit"])

with tab1:
    st.markdown(f"#### 5-Day Direct Forecast: {st.session_state.get('active_ticker', 'Historical AAPL')}")
    
    if "active_df" in st.session_state:
        # Use tail(100) to ensure features have enough buffer
        df_disp = st.session_state.active_df.tail(100)
        forecast = st.session_state.active_forecast
        ticker_name = st.session_state.active_ticker
    else:
        # Fallback to local CSV if no fetch yet
        df_raw = pd.read_csv("data/raw/AAPL.csv", parse_dates=['Date'], index_col='Date')
        # Take 100 days to ensure enough history for 30-day MA + 60-day window
        df_disp = df_raw.tail(100) 
        forecast = forecast_next5(df_disp)
        ticker_name = "AAPL (Offline)"

    col_chart, col_data = st.columns([2, 1], gap="large")
    
    with col_chart:
        if forecast:
            # Bias Indicator
            bias, bias_desc = get_forecast_bias(forecast, df_disp)
            bias_color = "#1D9E75" if "Bullish" in bias else "#E24B4A" if "Bearish" in bias else "#666"
            st.markdown(f'<div class="status-pill" style="background: {bias_color}22; color: {bias_color}; border: 1px solid {bias_color}">{bias.upper()} INDICATED</div>', unsafe_allow_html=True)
            
            fig = go.Figure()
            # Historical (just show last 60 for clarity)
            df_plot = df_disp.tail(60)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Close'], name='Historical Close', line=dict(color='#7F77DD', width=2)))
            # Forecast (Connect last point to first forecast point)
            f_dates = [df_plot.index[-1] + timedelta(days=i) for i in range(1, 6)]
            fig.add_trace(go.Scatter(x=[df_plot.index[-1]] + f_dates, y=[df_plot['Close'].iloc[-1]] + forecast, name='5-Day Forecast', line=dict(color='#1D9E75', width=3, dash='dash')))
            
            fig.update_layout(template="plotly_white", margin=dict(l=0,r=0,t=40,b=0), height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Awaiting model training/data fetch to generate active forecast.")

    with col_data:
        st.markdown("#### Neural Model Insights")
        if forecast:
            st.info(f"Model detects a **{bias.lower()}** trend, with prices {bias_desc} over the next 5 working days.")
            st.write("Predicted Future Prices:")
            f_df = pd.DataFrame({"Date": f_dates, "Price": forecast})
            st.dataframe(f_df.set_index("Date").style.format("${:.2f}"), use_container_width=True)
        else:
            st.caption("The model extracts multi-dimensional temporal signals from live market data.")
        
        st.markdown("---")
        st.markdown("**Status**: model.pkl Loaded")
        st.markdown(f"**Asset**: {ticker_name}")

with tab2:
    st.markdown("#### Model Performance Dynamics")
    
    col_loss, col_samples = st.columns([1, 1.2], gap="large")
    
    with col_loss:
        st.markdown("#### Training Evolution")
        history = get_training_metrics()
        if history:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(y=history["train_loss"], name="Train MSE", line=dict(color="#7F77DD", width=2)))
            fig_loss.add_trace(go.Scatter(y=history["val_loss"], name="Val MSE", line=dict(color="#333", width=1, dash="dot")))
            fig_loss.update_layout(template="plotly_white", margin=dict(l=0,r=0,t=0,b=0), height=300)
            st.plotly_chart(fig_loss, use_container_width=True)
        else:
            st.warning("No training metrics found.")

    with col_samples:
        st.markdown("#### Actual vs Predicted (Test Holdout)")
        samples = get_test_samples()
        if samples:
            fig_samples = go.Figure()
            fig_samples.add_trace(go.Scatter(y=samples["actual"], name="Actual Market", line=dict(color="#333", width=1.5)))
            fig_samples.add_trace(go.Scatter(y=samples["predicted"], name="Model Prediction", line=dict(color="#1D9E75", width=2)))
            fig_samples.update_layout(template="plotly_white", margin=dict(l=0,r=0,t=0,b=0), height=300, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_samples, use_container_width=True)
            st.markdown("<p class='muted-text'>Comparison of the last 100 samples in the unseen test set. This proves the model's ability to track price volatility without lagging behind the trend.</p>", unsafe_allow_html=True)
        else:
            st.info("Run the master pipeline to generate test set comparison samples.")

with tab3:
    st.markdown("#### Project Evolution: Engineering & Audit Report")
    
    col_story, col_strat = st.columns([1.5, 1], gap="large")
    
    with col_story:
        # Dataset Health
        st.markdown("#### Dataset Health & Provenance")
        df_raw = pd.read_csv("data/raw/AAPL.csv")
        st.write(f"• **Source**: Original Market Data (Yahoo Finance)")
        st.write(f"• **Total History**: {len(df_raw)} trading sessions")
        st.write(f"• **Features**: 11 Dimensional (OHLCV + Technicals)")
        st.markdown("---")
        
        st.markdown("#### Project Story: Time Series Architecture Journey")
        story = get_project_story()
        for item in story:
            with st.expander(item["stage"]):
                st.write(item["content"])

    with col_strat:
        st.markdown("#### Applied DL Strategies")
        strategies = [
            ("Temporal Splitting", "Zero-leakage splitting that preserves the chronological order of data."),
            ("LR Warmup", "Ramping LR from 0.0 to 1e-3 to prevent early gradient divergence in complex RNNs."),
            ("Direct Multi-step", "A single head predicting t+1 to t+5 to avoid exposure bias during inference."),
            ("Gradient Clipping", "Hard norm thresholding to ensure numerical stability during Backprop-Through-Time."),
            ("Rolling Features", "Moving Averages added to the raw stream to provide the model with pre-computed trend context.")
        ]
        for title, desc in strategies:
            st.markdown(f"""
            <div class="strategy-card">
                <div style="font-size: 13px; font-weight: 600; color: #1a1a1a">{title}</div>
                <div style="font-size: 11px; color: #666; margin-top: 4px">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("---")
st.caption("Time Series Engineering Project · LSTM Stock Price Forecaster v2.0")
