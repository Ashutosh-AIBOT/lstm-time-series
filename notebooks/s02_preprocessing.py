import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from notebooks.s01_eda import print_stop

# --- CONFIG ---
SEQ_LEN = 60
PRED_LEN = 5

def create_sequences(data, seq_len, pred_len, target_idx=0):
    """Generate 3D sliding window datasets."""
    X, y = [], []
    for i in tqdm(range(len(data) - seq_len - pred_len), desc="Creating windows"):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len, target_idx]) # Predict next close values
    return np.array(X), np.array(y)

# --- STOPS ---
def stop_4_temporal_split(df, auto_mode=False):
    print("\n[STOP 4] Temporal Dataset Split...")
    
    # 80/20 split based on time (no random shuffle!)
    split_idx = int(0.8 * len(df))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"Train Size: {len(train_df)}")
    print(f"Test Size: {len(test_df)} (Recent 20% of history)")
    
    explanation = (
        "• Random Split Leakage: Randomly shuffling data would allow the model to see future points in training.\n"
        "• Look-ahead Bias: Using next Tuesday to predict last Monday is invalid in time series.\n"
        "• Temporal Sequence: The test set must ALWAYS represent the most recent chronologically contiguous data."
    )
    print_stop(4, "Temporal Split", explanation, auto_mode)
    return train_df, test_df

def stop_5_minmax_scaling(train_df, test_df, auto_mode=False):
    print("\n[STOP 5] MinMaxScaler (Temporal-Safe)...")
    
    # Initialize Scaler
    scaler = MinMaxScaler()
    
    # Fit only on the training set to prevent leakage
    scaler.fit(train_df)
    
    # Transform both splits
    train_scaled = scaler.transform(train_df)
    test_scaled = scaler.transform(test_df)
    
    # Save the scaler for inference
    os.makedirs("models", exist_ok=True)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    explanation = (
        "• Fit vs Transform: Scaling parameters (min/max) are extracted ONLY from the training period.\n"
        "• Normalization: LSTM networks are numerically sensitive and converge much faster with [0, 1] inputs.\n"
        "• Target Scaling: We also scale the 'Close' target price to avoid huge gradient shifts from dollar units."
    )
    print_stop(5, "MinMax Scaling", explanation, auto_mode)
    return train_scaled, test_scaled, scaler

def stop_6_sliding_window(train_scaled, test_scaled, auto_mode=False):
    print("\n[STOP 6] Creating 3D Sliding Windows...")
    
    # Get target index (should be 0 for 'Close')
    # Assuming 'Close' is the first column for y extraction
    X_train, y_train = create_sequences(train_scaled, SEQ_LEN, PRED_LEN)
    X_test, y_test = create_sequences(test_scaled, SEQ_LEN, PRED_LEN)
    
    print(f"Train Shape X: {X_train.shape}, y: {y_train.shape}")
    print(f"Test Shape X: {X_test.shape}, y: {y_test.shape}")
    
    # Save for stage 3
    os.makedirs("data/processed", exist_ok=True)
    processed_data = {
        "train": (X_train, y_train),
        "test": (X_test, y_test)
    }
    with open("data/processed/sequences.pkl", "wb") as f:
        pickle.dump(processed_data, f)
    
    explanation = (
        "• sliding window: Look-back (60 days) -> Forecast (5 days).\n"
        "• 3D Tensors: [Batches, Timesteps, Features] is the standard RNN/LSTM input shape.\n"
        "• Multi-step: We are predicting five different targets simultaneously per sample.\n"
        "• Overlap: Consecutive windows overlap by 59 steps, which is normal for high-freq resolution."
    )
    print_stop(6, "Sliding Window", explanation, auto_mode)
    return X_train, y_train, X_test, y_test

def run_preprocessing(df, auto_mode=False):
    train_df, test_df = stop_4_temporal_split(df, auto_mode)
    train_scaled, test_scaled, scaler = stop_5_minmax_scaling(train_df, test_df, auto_mode)
    return stop_6_sliding_window(train_scaled, test_scaled, auto_mode)

if __name__ == "__main__":
    from notebooks.s01_eda import run_eda
    df = run_eda(auto_mode=True)
    if df is not None:
        run_preprocessing(df)
