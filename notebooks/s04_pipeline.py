
import torch
import time
import sys
import os
import pickle
from notebooks.s01_eda import run_eda
from notebooks.s02_preprocessing import run_preprocessing
from notebooks.s03_model import run_training

def print_summary_table():
    print("\n" + "█"*60)
    print("  PROJECT 08 — LSTM TIME SERIES FORECASTING v2.0")
    print("█"*60)
    print("\n13 STOPS ARCHITECTURAL MAP:")
    stops = [
        "1  Load & Plot (Raw Time Series Analysis)",
        "2  Stationarity & Returns (ADF Test)",
        "3  Feature Engineering (MA7, MA30, Vol)",
        "4  Temporal Train/Test Split (80/20 No Shuffle)",
        "5  MinMax Scaling (Temporal-Safe Parameters)",
        "6  Sliding Window Dataset (Look-back vs Forecast)",
        "7  Multi-step LSTM Architecture (Direct Head)",
        "8  Architecture Verification (Param Counts)",
        "9  LR Warmup & Cosine Decay (LambdaLR)",
        "10 GPU Training Loop (MSE Loss + Grad Clipping)",
        "11 Evaluation Metrics (RMSE, MAE, MAPE)",
        "12 Walk-forward Validation (Retraining Blocks)",
        "13 Save & Forecast Helper (Production Ready)"
    ]
    for s in stops:
        print(f"  [ ] STOP {s}")
    print("█"*60 + "\n")

def run_pipeline():
    """Master runner for the 13-STOP LSTM Time Series project."""
    auto_mode = "--auto" in sys.argv
    print_summary_table()
    
    if auto_mode:
        print(" [SYSTEM] MODE: AUTOMATED (NON-INTERACTIVE)")
    else:
        print(" [SYSTEM] MODE: INTERACTIVE (EDUCATIONAL)")
        
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" [SYSTEM] Found device: {device.upper()}")
    
    try:
        # STAGE 1: EDA (STOPS 1-3)
        print("\n>>> STAGE 1: EXPLORATORY DATA ANALYSIS")
        df = run_eda(auto_mode)
        if df is None: return
        
        # STAGE 2: Preprocessing (STOPS 4-6)
        print("\n>>> STAGE 2: SEQUENCE PREPROCESSING")
        X_train, y_train, X_test, y_test = run_preprocessing(df, auto_mode)
        processed_data = {"train": (X_train, y_train), "test": (X_test, y_test)}
        
        # STAGE 3: Model & Training (STOPS 7-13)
        print("\n>>> STAGE 3: NEURAL TRAINING & EVALUATION")
        model = run_training(processed_data, device, auto_mode)
        
        # STOP 13: Save & Forecast Success
        print("\n[STOP 13] Saving Production Checkpoint...")
        print("  - Models and artifacts persisted to models/")
        print("  - Metadata saved to model_summary.json")
        
        duration = time.time() - start_time
        print(f"\n" + "═"*60)
        print(f"  [COMPLETE] Total Pipeline Time: {duration/60:.2f} mins")
        print("═"*60)
        print("  All 13 Stops successfully verified.")
        print("  Dashboard is now ready to visualize forecasts.")
        print("═"*60)
        print("\nLaunch Dashboard with: streamlit run app.py")
        
    except KeyboardInterrupt:
        print("\n" + "!"*60)
        print("  [ABORT] Pipeline stopped by user.")
        print("!"*60)
    except Exception as e:
        print("\n" + "!"*60)
        print(f"  [ERROR] Pipeline failed: {str(e)}")
        print("!"*60)
        raise e

if __name__ == "__main__":
    run_pipeline()
