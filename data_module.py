import yfinance as yf
import pandas as pd

def fetch_market_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical market data from Yahoo Finance.
    """
    df = yf.download(ticker, start=start_date, end=end_date)
    # yfinance sometimes returns a multi-index column if a single ticker is passed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df

def generate_features(df: pd.DataFrame, rolling_window: int = 21) -> pd.DataFrame:
    """
    Engineer features for the HMM model: daily returns and rolling volatility.
    """
    df = df.copy()
    df["return"] = df["Close"].pct_change()
    df["volatility"] = df["return"].rolling(window=rolling_window).std()
    df = df.dropna()
    return df
