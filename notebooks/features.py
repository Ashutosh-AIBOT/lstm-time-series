import numpy as np
import pandas as pd

def prepare_features(df):
    """
    Standardized feature engineering for both training and inference.
    Expects OHLCV columns (at minimum Close).
    """
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # 1. Monthly (30-day) Returns
    df['Return'] = df['Close'].pct_change()
    
    # 2. Moving Averages
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA30'] = df['Close'].rolling(window=30).mean()
    
    # 3. Volatility (Rolling Std)
    df['Volatility'] = df['Close'].rolling(window=7).std()
    
    # 4. Technical Indicators (Simple RSI-like)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Final cleanup - ensure columns are in the EXACT order the scaler expects
    # The order must match: Date, Close, High, Low, Open, Volume, Return, MA7, MA30, Volatility, RSI
    # Since Date is normally the index, we ensure OHLCV + Features
    
    # Assuming CSV structure is Date, Close, High, Low, Open, Volume
    # We will return the dataframe with all generated features.
    return df
