"""Case 3: Download stock market data from Yahoo Finance."""

import os
import yfinance as yf

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Download Apple stock data (well-known, high-volume ticker)
ticker = "AAPL"
print(f"Downloading {ticker} stock data from Yahoo Finance...")

stock = yf.download(ticker, start="2020-01-01", end="2025-01-01", auto_adjust=True)
stock.to_csv(os.path.join(DATA_DIR, f"{ticker}.csv"))

print(f"Saved {len(stock)} rows to {DATA_DIR}/{ticker}.csv")
print(stock.head())
