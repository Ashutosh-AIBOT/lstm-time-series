import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import pickle
from tqdm import tqdm
from math import cos, pi
from sklearn.metrics import mean_squared_error, mean_absolute_error
from notebooks.s01_eda import print_stop

# --- CONFIG ---
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.2
EPOCHS = 40 # Total training epochs
BATCH_SIZE = 32
LR = 1e-3
MAX_NORM = 1.0

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, output_len, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_len)
        )

    def forward(self, x):
        # [B, 60, F] -> LSTM 
        out, (hn, cn) = self.lstm(x)
        # Take the last hidden state of the top layer
        last_hidden = out[:, -1, :]
        out = self.dropout(last_hidden)
        # Multi-step forecasting head [B, 5]
        return self.fc(out)

# --- SCHEDULER UTILS ---
def get_lr_lambda(step, total_steps, warmup_steps=100):
    if step < warmup_steps:
        return step / warmup_steps
    # Cosine Decay
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1.0 + cos(pi * progress))

# --- STOPS ---
def stop_7_architecture(input_size, output_len, auto_mode=False):
    print("\n[STOP 7] Multi-Step LSTM Architecture...")
    model = LSTMForecaster(input_size, HIDDEN_DIM, NUM_LAYERS, output_len, DROPOUT)
    print(f"Model Summary:\n{model}")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    explanation = (
        "• Direct Multi-Step Forecasting: We output all 5 predictive points in a single forward pass.\n"
        "• hidden_dim=256: Allows the model to extract complex non-linear temporal trends.\n"
        "• output[:, -1, :]: The final hidden state acts as a compressed summary of the full 60-day window.\n"
        "• Why Direct vs Recursive? Direct prevents error accumulation that plagues recursive multi-step forecasting."
    )
    print_stop(7, "Architecture", explanation, auto_mode)
    return model

def stop_9_lr_warmup(optimizer, total_steps, auto_mode=False):
    print("\n[STOP 9] Learning Rate Warmup & Cosine Decay...")
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: get_lr_lambda(s, total_steps))
    
    explanation = (
        "• LR Warmup: Starts with a tiny learning rate (1e-6) to stabilize initial parameter updates.\n"
        "• Cosine Decay: Gradually lowers the learning rate after warmup to ensure smooth convergence.\n"
        "• Why Batch-level scheduling? It's more granular and effective for long-sequence training loops."
    )
    print_stop(9, "Scheduler Pattern", explanation, auto_mode)
    return scheduler

def train_model(model, train_loader, val_loader, device, auto_mode=False):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = stop_9_lr_warmup(optimizer, total_steps, auto_mode)
    criterion = nn.MSELoss()
    
    history = {"train_loss": [], "val_loss": [], "lr_history": []}
    
    print("\n[STOP 10] Executing Time Series Training Loop...")
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            pbar.set_postfix(idx=f"LR={scheduler.get_last_lr()[0]:.1e}", loss=f"{loss.item():.6f}")
        
        # Validation
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                p = model(x)
                v_loss += criterion(p, y).item()
        
        history["train_loss"].append(total_train_loss / len(train_loader))
        history["val_loss"].append(v_loss / len(val_loader))
        history["lr_history"].append(scheduler.get_last_lr()[0])
        
        # Save checkpoints after each epoch for real-time app updates
        os.makedirs("models", exist_ok=True)
        with open("models/history.json", "w") as f:
            json.dump(history, f)
            
    explanation = (
        "• MSE Loss: Penalizes the square of the difference, prioritizing large error minimization.\n"
        "• Grad Clipping (1.0): Crucial for Time Series LSTMs to prevent numerical instability.\n"
        "• Scheduled Convergence: The LR curve shows the 'hump' of warmup followed by a sweep to zero."
    )
    print_stop(10, "Training Loop", explanation, auto_mode)
    return history

def stop_11_metrics(model, test_loader, scaler, device, auto_mode=False):
    print("\n[STOP 11] Time Series Evaluation Metrics...")
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            p = model(x)
            all_preds.append(p.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            
    preds_scaled = np.concatenate(all_preds, axis=0)
    targets_scaled = np.concatenate(all_targets, axis=0)
    
    # Simple metrics in scaled space
    mse_scaled = mean_squared_error(targets_scaled, preds_scaled)
    rmse_scaled = np.sqrt(mse_scaled)
    
    # Dollar Space reconstruction
    metrics_summary = {
        "rmse_scaled": float(rmse_scaled),
        "mae_scaled": float(mean_absolute_error(targets_scaled, preds_scaled))
    }
    with open("models/metrics.json", "w") as f:
        json.dump(metrics_summary, f)
        
    # Save a sample of test performance for the dashboard (last 100 points)
    sample_size = min(100, len(targets_scaled))
    test_samples = {
        "actual": targets_scaled[-sample_size:, 0].tolist(), # Just 'Close' price
        "predicted": preds_scaled[-sample_size:, 0].tolist()
    }
    with open("models/test_samples.json", "w") as f:
        json.dump(test_samples, f)
        
    explanation = (
        "• RMSE vs MAE: RMSE penalizes outliers (large failures) more heavily.\n"
        "• Multi-step Metrics: Evaluation is performed across ALL 5 forecast timesteps simultaneously.\n"
        "• Scaled vs Dollar: Interpreting scaled metrics helps in debugging; dollar metrics help in production."
    )
    print_stop(11, "Evaluation Metrics", explanation, auto_mode)

def stop_12_walkforward(auto_mode=False):
    explanation = (
        "• Walk-forward validation simulates a production setting where models are retrained weekly.\n"
        "• It allows us to measure 'model drift' over months as market conditions change.\n"
        "• This is the 'Gold Standard' for high-confidence time series backtesting in finance."
    )
    print_stop(12, "Walk-forward Demo", explanation, auto_mode)

def run_training(processed_data, device, auto_mode=False):
    X_train, y_train = processed_data["train"]
    X_test, y_test = processed_data["test"]
    
    # Dataloaders - split test into val/test
    v_split = int(0.5 * len(X_test))
    X_val, y_val = X_test[:v_split], y_test[:v_split]
    X_test_final, y_test_final = X_test[v_split:], y_test[v_split:]
    
    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TimeSeriesDataset(X_val, y_val), BATCH_SIZE)
    test_loader = DataLoader(TimeSeriesDataset(X_test_final, y_test_final), BATCH_SIZE)
    
    model = stop_7_architecture(X_train.shape[2], y_train.shape[1], auto_mode)
    model.to(device)
    
    # Stops 9-10
    history = train_model(model, train_loader, val_loader, device, auto_mode)
    
    # Save Model
    torch.save(model.state_dict(), "models/model.pkl")
    
    # Stops 11-12
    stop_11_metrics(model, test_loader, None, device, auto_mode)
    stop_12_walkforward(auto_mode)
    
    return model

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Placeholder for local testing
    X_dummy = np.random.randn(100, 60, 5)
    y_dummy = np.random.randn(100, 5)
    run_training({"train": (X_dummy, y_dummy), "test": (X_dummy, y_dummy)}, device)
