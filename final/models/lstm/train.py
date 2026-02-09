"""
LSTM model for ActivePower time-series prediction.

Loads preprocessed train/test data, trains an LSTM network,
evaluates performance, and exports the model and metrics.
"""

import os
import sys
import json
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RANDOM_STATE = 20940
WINDOW_SIZE = 60  # 60-minute lookback window
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2

TARGET = "ActivePower"
FEATURE_COLUMNS = [
    "PowerA",
    "PowerB",
    "PowerC",
    "ReactivePower",
    "CurrentHolderPositionA",
    "CurrentHolderPositionB",
    "CurrentHolderPositionC",
    "GasPressureUnderFurnaceA",
    "AirTemperatureMantelB",
    "FurnacePodTemparature",
]

# Paths (relative to project root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Reproducibility
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TimeSeriesDataset(Dataset):
    """Sliding-window dataset for time-series forecasting."""

    def __init__(self, features: np.ndarray, targets: np.ndarray, window_size: int):
        self.features = features
        self.targets = targets
        self.window_size = window_size

    def __len__(self):
        return len(self.features) - self.window_size

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.window_size]
        y = self.targets[idx + self.window_size]
        return torch.FloatTensor(x), torch.FloatTensor([y])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, dropout: float
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take the output from the last time step
        last_hidden = lstm_out[:, -1, :]
        out = self.fc(last_hidden)
        return out


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            total_loss += loss.item()
            n_batches += 1
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    avg_loss = total_loss / n_batches
    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()
    return avg_loss, all_preds, all_targets


def main():
    print(f"Device: {DEVICE}")
    print(f"Window size: {WINDOW_SIZE}, Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")
    print(f"Hidden size: {HIDDEN_SIZE}, Layers: {NUM_LAYERS}, Dropout: {DROPOUT}")
    print()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    # Separate features and target
    X_train = train_df[FEATURE_COLUMNS].values.astype(np.float32)
    y_train = train_df[TARGET].values.astype(np.float32)
    X_test = test_df[FEATURE_COLUMNS].values.astype(np.float32)
    y_test = test_df[TARGET].values.astype(np.float32)

    # ------------------------------------------------------------------
    # 2. Scale data (fit on train only)
    # ------------------------------------------------------------------
    feature_scaler = MinMaxScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)

    target_scaler = MinMaxScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

    # ------------------------------------------------------------------
    # 3. Create datasets and dataloaders
    # ------------------------------------------------------------------
    train_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled, WINDOW_SIZE)
    test_dataset = TimeSeriesDataset(X_test_scaled, y_test_scaled, WINDOW_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train sequences: {len(train_dataset)}, Test sequences: {len(test_dataset)}")

    # ------------------------------------------------------------------
    # 4. Build model
    # ------------------------------------------------------------------
    model = LSTMModel(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print()

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    train_losses = []
    test_losses = []

    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        test_loss, _, _ = evaluate(model, test_loader, criterion)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{EPOCHS}  |  Train Loss: {train_loss:.6f}  |  Test Loss: {test_loss:.6f}"
            )

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.1f}s")

    # ------------------------------------------------------------------
    # 6. Final evaluation (inverse-transform predictions to original scale)
    # ------------------------------------------------------------------
    _, preds_scaled, targets_scaled = evaluate(model, test_loader, criterion)

    preds_original = target_scaler.inverse_transform(
        preds_scaled.reshape(-1, 1)
    ).flatten()
    targets_original = target_scaler.inverse_transform(
        targets_scaled.reshape(-1, 1)
    ).flatten()

    mae = mean_absolute_error(targets_original, preds_original)
    mse = mean_squared_error(targets_original, preds_original)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets_original, preds_original)

    print(f"\n--- Test Metrics (original scale) ---")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

    # ------------------------------------------------------------------
    # 7. Export model and scalers
    # ------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model_path = os.path.join(OUTPUT_DIR, "lstm_model.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "input_size": len(FEATURE_COLUMNS),
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
            },
        },
        model_path,
    )
    print(f"\nModel saved to {model_path}")

    scaler_path = os.path.join(OUTPUT_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(
            {"feature_scaler": feature_scaler, "target_scaler": target_scaler}, f
        )
    print(f"Scalers saved to {scaler_path}")

    # ------------------------------------------------------------------
    # 8. Save results JSON
    # ------------------------------------------------------------------
    results = {
        "model": "lstm",
        "target": TARGET,
        "features": FEATURE_COLUMNS,
        "window_size": WINDOW_SIZE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "random_state": RANDOM_STATE,
        "device": str(DEVICE),
        "total_parameters": total_params,
        "training_time_seconds": round(training_time, 2),
        "metrics": {
            "mae": round(float(mae), 6),
            "mse": round(float(mse), 6),
            "rmse": round(float(rmse), 6),
            "r2": round(float(r2), 6),
        },
        "training_history": {
            "train_loss": [round(float(l), 6) for l in train_losses],
            "test_loss": [round(float(l), 6) for l in test_losses],
        },
    }

    results_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
