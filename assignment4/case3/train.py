"""Case 3: Time Series Forecasting for Stock Market Prediction using Prophet."""

import json
import os
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = "data"
TICKER = "AAPL"
DATA_PATH = os.path.join(DATA_DIR, f"{TICKER}.csv")
RESULTS_PATH = "results.json"

# ── Load data ─────────────────────────────────────────────────────────────────
# yfinance saves with 3 header rows (Price/Ticker/Date), skip first 2
df = pd.read_csv(DATA_PATH, skiprows=[1, 2])
df.rename(columns={"Price": "Date"}, inplace=True)

# Prophet expects columns: ds (date) and y (target)
prophet_df = pd.DataFrame()
prophet_df["ds"] = pd.to_datetime(df["Date"])
prophet_df["y"] = df["Close"].astype(float)
prophet_df = prophet_df.dropna().sort_values("ds").reset_index(drop=True)

# ── Train/test split (last 60 trading days as test) ──────────────────────────
test_days = 60
train_df = prophet_df.iloc[:-test_days]
test_df = prophet_df.iloc[-test_days:]

# ── Train Prophet ─────────────────────────────────────────────────────────────
model = Prophet(
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True,
    changepoint_prior_scale=0.05,
)
model.fit(train_df)

# ── Forecast ──────────────────────────────────────────────────────────────────
future = model.make_future_dataframe(periods=test_days)
forecast = model.predict(future)

# Merge actuals with predictions on test period
forecast_test = forecast[forecast["ds"].isin(test_df["ds"])][["ds", "yhat", "yhat_lower", "yhat_upper"]]
merged = test_df.merge(forecast_test, on="ds")

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_true = merged["y"].values
y_pred = merged["yhat"].values

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

results = {
    "case": "Time Series Forecasting for Stock Market Prediction",
    "model": "Prophet (Facebook)",
    "dataset": f"Yahoo Finance – {TICKER}",
    "dataset_shape": {"rows": len(prophet_df), "features": 2},
    "date_range": {
        "start": str(prophet_df["ds"].min().date()),
        "end": str(prophet_df["ds"].max().date()),
    },
    "train_size": len(train_df),
    "test_size": len(test_df),
    "metrics": {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2_score": round(r2, 4),
        "mape_percent": round(mape, 4),
    },
    "sample_predictions": [
        {
            "date": str(row["ds"].date()),
            "actual": round(float(row["y"]), 2),
            "predicted": round(float(row["yhat"]), 2),
        }
        for _, row in merged.head(10).iterrows()
    ],
    "hyperparameters": {
        "daily_seasonality": False,
        "weekly_seasonality": True,
        "yearly_seasonality": True,
        "changepoint_prior_scale": 0.05,
    },
}

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {RESULTS_PATH}")
print(f"MAE:   {results['metrics']['mae']}")
print(f"RMSE:  {results['metrics']['rmse']}")
print(f"R²:    {results['metrics']['r2_score']}")
print(f"MAPE:  {results['metrics']['mape_percent']}%")
