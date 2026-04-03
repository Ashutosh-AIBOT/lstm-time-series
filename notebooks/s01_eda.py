import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm
import time
from notebooks.features import prepare_features

# --- SHARED UTILS ---
def print_stop(stop_num, title, explanation, auto_mode=False):
    """Shared pretty-printer for pipeline segments."""
    print("\n" + "═"*60)
    print(f"  STOP {stop_num} — {title.upper()}")
    print("═"*60)
    print(f"\nEXPLANATION:\n{explanation}")
    print("─"*60)
    if not auto_mode:
        input("\n[Action] Press Enter to continue and execute this stop...")
    else:
        print(f"\n[Auto Mode] Proceeding from STOP {stop_num}...")
        time.sleep(1)

# --- STOPS ---
def stop_1_load_data(path, auto_mode=False):
    print("\n[STOP 1] Loading Time Series Data (AAPL)...")
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    
    print(f"Dataset Size: {len(df)} days")
    print(f"Date Range: {df.index.min().date()} to {df.index.max().date()}")
    print("\nBasic Stats (Close Price):")
    print(df['Close'].describe())
    
    # Plotting
    os.makedirs("charts", exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], color='#7F77DD', label='Close Price')
    plt.title("AAPL Close Price History")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.savefig("charts/ts_raw.png")
    print("Chart saved to charts/ts_raw.png")
    
    explanation = (
        "• Time series data is a sequence of observations ordered chronologically.\n"
        "• Temporal Dependency: Unlike tabular data, current rows are biased by past states.\n"
        "• Forecasting Objective: Predicting the future state based on historical patterns.\n"
        "• AAPL stock shows a clear upward trend, indicating it is non-stationary."
    )
    print_stop(1, "Load & Plot", explanation, auto_mode)
    return df

def stop_2_stationarity(df, auto_mode=False):
    print("\n[STOP 2] Analyzing Stationarity (ADF Test)...")
    
    # Compute Returns for Stationarity
    df['Return'] = df['Close'].pct_change().dropna()
    
    # ADF Test on Raw Price
    result_raw = adfuller(df['Close'])
    # ADF Test on Returns
    result_ret = adfuller(df['Return'].dropna())
    
    print(f"ADF p-value (Raw Close): {result_raw[1]:.4f}")
    print(f"ADF p-value (Returns): {result_ret[1]:.4e}")
    
    explanation = (
        "• Stationarity: Mean and variance remain constant over time.\n"
        "• ADF Test: p-value < 0.05 suggests the series is stationary.\n"
        "• Most raw stock prices are non-stationary; returns are typically much easier to model.\n"
        "• LSTM models often benefit from the relative stability of returns/log-prices."
    )
    print_stop(2, "Stationarity & Returns", explanation, auto_mode)

def stop_3_feature_engineering(df, auto_mode=False):
    print("\n[STOP 3] Temporal Feature Engineering...")
    
    df = prepare_features(df)
    
    df.dropna(inplace=True)
    print(f"Final Feature Set: {df.columns.tolist()}")
    print(f"Remaining Data Points: {len(df)}")
    
    explanation = (
        "• Rolling features (MA7, MA30) capture short-term and long-term trends.\n"
        "• Volatility indices provide the model with a sense of market risk/stability.\n"
        "• Indicators like RSI give the LSTM 'pre-processed' semantic signals.\n"
        "• Data Leakage Avoidance: Rolling windows only look backward in time."
    )
    print_stop(3, "Feature Engineering", explanation, auto_mode)
    return df

def run_eda(auto_mode=False):
    data_path = "data/raw/AAPL.csv"
    if not os.path.exists(data_path):
        print(f"[ERROR] {data_path} not found. Please ensure AAPL.csv exists.")
        return None
        
    df = stop_1_load_data(data_path, auto_mode)
    stop_2_stationarity(df, auto_mode)
    final_df = stop_3_feature_engineering(df, auto_mode)
    return final_df

if __name__ == "__main__":
    run_eda()
