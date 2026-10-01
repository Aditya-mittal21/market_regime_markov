import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_sharpe(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    excess_ret = returns - risk_free_rate / 252
    return np.sqrt(252) * (excess_ret.mean() / excess_ret.std())

def run_backtest(df: pd.DataFrame, return_col: str = "return", label_col: str = "regime_label"):
    """
    Simulate a strategy that goes long only when the regime is 'bull'.
    """
    df = df.copy()
    df["position"] = (df[label_col] == "bull").astype(int)
    # shift position by 1 day to avoid lookahead bias
    df["strategy_ret"] = df["position"].shift(1) * df[return_col]
    
    df = df.dropna()
    
    df["market_cum_return"] = (1 + df[return_col]).cumprod()
    df["strategy_cum_return"] = (1 + df["strategy_ret"]).cumprod()
    
    market_sharpe = calculate_sharpe(df[return_col])
    strategy_sharpe = calculate_sharpe(df["strategy_ret"])
    
    print(f"Market Sharpe Ratio: {market_sharpe:.4f}")
    print(f"Strategy Sharpe Ratio: {strategy_sharpe:.4f}")
    
    print(f"Market Final Cum Return: {df['market_cum_return'].iloc[-1]:.4f}")
    print(f"Strategy Final Cum Return: {df['strategy_cum_return'].iloc[-1]:.4f}")
    
    return df

def plot_results(df: pd.DataFrame):
    """
    Plot the cumulative returns of the strategy vs market.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["market_cum_return"], label="Market (Buy & Hold)")
    plt.plot(df.index, df["strategy_cum_return"], label="Regime Strategy")
    plt.legend()
    plt.title("Backtest: Market vs HMM Regime Strategy")
    plt.grid(True)
    plt.show()
