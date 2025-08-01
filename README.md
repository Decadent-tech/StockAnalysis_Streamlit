---
title: Stock Analyzer Dashboard
emoji: 📈
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: "1.28.0"
app_file: app.py
pinned: false
---

# Stock Analyzer Dashboard

An interactive **Streamlit** app for analyzing, forecasting, and comparing stock market data using machine learning, time series models, and technical indicators.

## Features

### Price & Technical Analysis
- Historical stock data from Yahoo Finance
- Daily returns, 7-day & 21-day Moving Averages
- Volatility estimation using rolling standard deviation
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)

###  ML Forecasting
- **Linear Regression** with lagged Close prices
- **SARIMAX** (Seasonal ARIMA with exogenous variables) for time-series forecasting

### Clustering
- **KMeans** clustering of stocks based on Close & Volume
- **DBSCAN** for anomaly and density-based clustering

### Economic Integration
- FRED API for fetching and overlaying **US CPI** data with stock trends

### Multi-Ticker Comparison
- Compare multiple tickers side-by-side
- Correlation heatmap between stocks
- Downloadable multi-ticker dataset as CSV

### Portfolio Simulator
- Allocate weights across tickers
- Simulate and visualize cumulative portfolio returns

---

## How to Use

1. Enter a valid stock ticker (e.g., `TCS`, `RELIANCE`, `AAPL`, etc.)
2. Choose a valid start and end date
3. Optionally enter comparison tickers
4. Explore each tab:
   - Price metrics and trends
   - Predictions
   - Technicals
   - Economic overlays
   - Clustering and portfolio simulation

---

## Tech Stack

| Category         | Libraries / Tools                         |
|------------------|--------------------------------------------|
| UI & App         | `Streamlit`, `matplotlib`, `seaborn`       |
| Data Retrieval   | `yfinance`, `pandas_datareader`            |
| Feature Engg     | `pandas`, `numpy`, `ta`                    |
| ML Models        | `scikit-learn`, `statsmodels`              |
| Time & Location  | `datetime`, `pytz`                         |
| Clustering       | `KMeans`, `DBSCAN`                         |

---

## Project Structure

```
app.py                 # Main Streamlit dashboard
requirements.txt       # All required dependencies
README.md              # Project + Hugging Face config
```

---

## Try It on Hugging Face

> [![HuggingFace](https://img.shields.io/badge/View%20App-HuggingFace-blue?logo=huggingface)](https://huggingface.co/spaces/riya1chatterjee/StockAnalyzer)

---

## Author

Made by a Data Science enthusiast using real-time stock data and machine learning techniques.  
_This app is for educational and demo purposes only — not financial advice._
