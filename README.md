# 📊 Stock Analysis & Forecasting Dashboard

An interactive Streamlit app for analyzing, visualizing, clustering, and forecasting stock market trends using machine learning, time series models, technical indicators, and macroeconomic data.

---

## 🚀 Features

### 1. **📈 Stock Data Retrieval**
- Pulls historical stock data from **Yahoo Finance** using `yfinance`
- Supports Indian (`.NS`, `.BO`) and international tickers
- Flexible start/end date selection

### 2. **📉 Technical Analysis & Visualization**
- Daily closing price visualization
- Moving Averages (7-day & 21-day)
- Bollinger Bands
- Volatility (21-day rolling STD)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)

### 3. **📊 Forecasting Models**
- **Linear Regression** using lag-based features
- **SARIMAX** for time-series forecasting (30-day)

### 4. **🧠 Unsupervised Clustering**
- **KMeans** clustering using Close & Volume
- **DBSCAN** for anomaly or density-based grouping

### 5. **🌐 Macroeconomic Overlay**
- US CPI data integration from FRED
- Compare macroeconomic trends with stock price movement

### 6. **📌 Prediction Comparison Dashboard**
- Compare Linear Regression vs. SARIMAX predictions
- Evaluation Metrics: R², MAE, RMSE, MSE

### 7. **📊 Portfolio Simulator**
- Multi-ticker input with custom weight sliders
- Simulates cumulative returns from selected tickers

### 8. **⏱️ Real-Time Market Status**
- Auto-detects ticker region (India or US)
- Displays whether market is currently **open or closed** using timezone logic

---

## 📸 Demo



##  Use Cases
- Analyze stock performance with technical indicators
- Predict future stock prices using ML and time-series models
- Identify stock clusters for strategy design
- Simulate portfolios with multiple global stocks
- Compare predictive models under different market conditions
- Monitor stock behavior during open vs. closed markets

## Technical Indicators 

| Category             | Tools & Libraries                              |
| -------------------- | ---------------------------------------------- |
| Data Collection      | `yfinance`, `pandas_datareader`                |
| Data Processing      | `pandas`, `numpy`, rolling statistics          |
| Visualization        | `Streamlit`, `matplotlib`, `seaborn`           |
| Technical Indicators | `ta` (RSI, MACD, Bollinger Bands)              |
| ML Models            | `scikit-learn` (Linear Regression, Clustering) |
| Time Series Forecast | `statsmodels` (SARIMAX)                        |
| Macroeconomic Data   | FRED (US CPI) via `pandas_datareader`          |
| Real-Time Logic      | `pytz`, `datetime`                             |


## Installation 

# Clone the repository
git clone 
cd stock-analysis-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run stockanalyser.py


