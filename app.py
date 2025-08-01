import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import ta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from datetime import date, timedelta
from datetime import datetime, time
import pytz

# ---------- Helper to Normalize Tickers for Indian Stocks ---------- #
def get_valid_ticker(ticker):
    indian_stocks = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN", "ICICIBANK", "WIPRO", "ITC", "COALINDIA", "ADANIENT"]
    ticker = ticker.strip().upper()
    if ticker in indian_stocks:
        return ticker + ".NS"  # NSE by default
    return ticker

def is_market_open(ticker):
    now_utc = datetime.now(pytz.utc)

    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        tz = pytz.timezone("Asia/Kolkata")
        now_local = now_utc.astimezone(tz)
        open_time = time(9, 15)
        close_time = time(15, 30)
        market = "NSE/BSE India"
    else:
        tz = pytz.timezone("US/Eastern")
        now_local = now_utc.astimezone(tz)
        open_time = time(9, 30)
        close_time = time(16, 0)
        market = "US (NYSE/NASDAQ)"

    is_open = open_time <= now_local.time() <= close_time and now_local.weekday() < 5

    return {
        "market": market,
        "is_open": is_open,
        "local_time": now_local.strftime("%Y-%m-%d %H:%M:%S"),
    }

def resolve_ticker(ticker: str):
    ticker = ticker.strip().upper()
    if '.' in ticker:
        return ticker

    for suffix in [".NS", ".BO", ""]:
        try:
            test_ticker = yf.Ticker(ticker + suffix)
            hist = test_ticker.history(period="5d")
            if not hist.empty:
                return ticker + suffix
        except:
            continue
    return None

# ----------------- Sidebar Inputs -----------------
with st.sidebar:
    st.title(" Stock Dashboard")
    txt_raw = st.text_input("Enter Main Stock Ticker (e.g., TCS, AAPL, RELIANCE)", "TCS")
    txt = resolve_ticker(txt_raw)
    if not txt:
        st.error("Could not resolve the entered ticker. Please try another.")
        st.stop()

    tickers_input = st.text_input("Compare Stocks (comma-separated)", "TCS, AAPL, RELIANCE")
    tickers = [get_valid_ticker(t) for t in tickers_input.split(',') if t.strip()]


    if not tickers:
        st.warning("No valid comparison tickers found.")

    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")

    if start_date >= end_date:
        st.error("Start Date must be before End Date.")
        st.stop()

    st.sidebar.write("Resolved Main Ticker:", txt)
    st.sidebar.write("Resolved Comparison Tickers:", tickers)
    status = is_market_open(txt)
    status_msg = "\U0001F7E2 Market is OPEN" if status["is_open"] else "\U0001F534 Market is CLOSED"

# ----------------- Fetch Stock Data -----------------
dat = yf.Ticker(txt)
ticker_df = dat.history(start=start_date, end=end_date)
if ticker_df.empty:
    st.warning("No stock data found. Try a different ticker or wider date range.")
    st.stop()

ticker_df['Daily Return'] = ticker_df['Close'].pct_change()
ticker_df['MA7'] = ticker_df['Close'].rolling(window=7).mean()
ticker_df['MA21'] = ticker_df['Close'].rolling(window=21).mean()
ticker_df['Volatility'] = ticker_df['Close'].rolling(window=21).std()
ticker_df['Close_Lag1'] = ticker_df['Close'].shift(1)
ticker_df['Close_Lag2'] = ticker_df['Close'].shift(2)
ticker_df['Close_Lag3'] = ticker_df['Close'].shift(3)
ticker_df.dropna(inplace=True)

# Ensure sufficient data
if ticker_df.empty or len(ticker_df) < 10:
    st.warning("Not enough data to train the model. Extend the date range.")
    st.stop()

# ----------------- Linear Regression Model -----------------
X = ticker_df[['Close_Lag1', 'Close_Lag2', 'Close_Lag3']]
y = ticker_df['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=y_test.index)

# ----------------- Technical Indicators -----------------
ticker_df['RSI'] = ta.momentum.RSIIndicator(ticker_df['Close']).rsi()
ticker_df['MACD'] = ta.trend.MACD(ticker_df['Close']).macd()

# ----------------- Tabs Layout -----------------
tab1, tab2, tab3, tab4, tab5, tab6 , tab7 = st.tabs([
    "📈 Price & Metrics", "📘 Linear Regression", 
    "📊 Technicals", "📉 Economic", 
    "🧩 Clustering & DBSCAN", "🔮 SARIMAX" ,"📌 Multi-Ticker Comparison"
])

with tab1:
    st.subheader(f"{txt.upper()} Stock Data")
    st.dataframe(ticker_df.tail(10))
    st.subheader(f"{txt.upper()} Overview ")
    st.line_chart(ticker_df['Close'])

with tab2:
    st.subheader("Predicted vs Actual Closing Price")
    st.line_chart(results)
    st.metric("R² Score", round(r2_score(y_test, y_pred), 2))
    st.write("MAE:", mean_absolute_error(y_test, y_pred))
    st.write("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    st.write("MSE:", mean_squared_error(y_test, y_pred))

with tab3:
    st.subheader("RSI and MACD")
    st.line_chart(ticker_df[['RSI', 'MACD']])
    st.subheader("Volatility (21-day Rolling STD)")
    st.line_chart(ticker_df['Volatility'])

with tab4:
    st.subheader("Macroeconomic Indicator: US CPI")
    cpi = web.DataReader('CPIAUCSL', 'fred', start_date, end_date)
    ticker_df.index = pd.to_datetime(ticker_df.index).tz_localize(None)
    cpi.index = pd.to_datetime(cpi.index).tz_localize(None)
    
    st.line_chart(cpi)
    combined = pd.merge(ticker_df.reset_index()[['Date', 'Close']], cpi, left_on='Date', right_index=True, how='inner')
    st.subheader("Close Price vs US CPI")
    st.line_chart(combined.set_index('Date'))

with tab5:
    st.header("KMeans & DBSCAN Clustering")

    if 'Close' in ticker_df.columns and 'Volume' in ticker_df.columns:
        clustering_data = ticker_df[['Close', 'Volume']].dropna()

        if len(clustering_data) >= 5:
            # KMeans
            kmeans = KMeans(n_clusters=3, random_state=42)
            clustering_data['KMeans'] = kmeans.fit_predict(clustering_data)

            # DBSCAN
            dbscan = DBSCAN(eps=0.05, min_samples=3)
            clustering_data['DBSCAN'] = dbscan.fit_predict(clustering_data)

            st.subheader("KMeans Clustering on Close & Volume")
            st.dataframe(clustering_data[['Close', 'Volume', 'KMeans']].tail(10))
            st.subheader("DBSCAN Clustering on Close & Volume")
            st.dataframe(clustering_data[['Close', 'Volume', 'DBSCAN']].tail(10))
        else:
            st.warning("Not enough data for clustering.")
    else:
        st.warning("Required fields for clustering not found.")

with tab6:
    st.info("This model uses historical close prices to forecast the next 30 days.")
    ts_data = ticker_df['Close'].dropna()
    ts_data.index = pd.to_datetime(ts_data.index)

    try:
        model = SARIMAX(ts_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                        enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=30)
        forecast.index = pd.date_range(start=ts_data.index[-1] + pd.Timedelta(days=1), periods=30)
        forecast_df = pd.DataFrame({'Forecast': forecast})
        st.subheader("30-Day SARIMAX Forecast")
        st.line_chart(pd.concat([ts_data.tail(60), forecast_df]))
        st.dataframe(forecast_df)
    except Exception as e:
        st.error(f"Error fitting SARIMAX: {e}")

with tab7:
    st.subheader("Multi-Ticker Close Price Comparison")

    multi_df = pd.DataFrame()

    for ticker in tickers:
        try:
            t_data = yf.Ticker(ticker).history(start=start_date, end=end_date)
            close_series = t_data['Close'].rename(ticker)
            multi_df = pd.concat([multi_df, close_series], axis=1)
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")

    if not multi_df.empty:
        st.line_chart(multi_df)
        st.dataframe(multi_df.tail())

        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(multi_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)


        # ------------------ Portfolio Simulation ------------------ #
        st.subheader("💼 Portfolio Simulator")

        weights = []
        for ticker in tickers:
            w = st.slider(f"Weight for {ticker} (%)", min_value=0, max_value=100, value=round(100 / len(tickers)))
            weights.append(w)

        total_weight = sum(weights)
        if total_weight != 100:
            st.error("Total weight must equal 100%. Adjust sliders.")
        else:
            norm_weights = np.array(weights) / 100
            daily_returns = multi_df.pct_change().dropna()
            portfolio_returns = (daily_returns * norm_weights).sum(axis=1)
            cumulative_returns = (1 + portfolio_returns).cumprod()

            st.line_chart(cumulative_returns)
            st.metric("Total Return (%)", f"{(cumulative_returns[-1] - 1)*100:.2f}")
        # Downloadable CSV
        csv = multi_df.to_csv(index=True).encode('utf-8')
        st.download_button("Download Multi-Ticker Data CSV", csv, "multi_ticker_data.csv", "text/csv")
    else:
        st.warning("Multi-ticker data is empty. Check ticker symbols or date range.")
