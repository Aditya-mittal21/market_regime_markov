from data_module import fetch_market_data, generate_features
from hmm_classifier import MarketRegimeHMM
from backtester import run_backtest, plot_results

def main():
    ticker = "SPY"
    start_date = "2020-01-01"
    end_date = "2025-08-01"
    
    print(f"Fetching data for {ticker}...")
    df = fetch_market_data(ticker, start_date, end_date)
    
    print("Engineering features...")
    df = generate_features(df, rolling_window=21)
    
    print("Training HMM Regime Model...")
    hmm_model = MarketRegimeHMM(n_states=2, n_iter=100)
    df_classified = hmm_model.fit_predict(df, feature_cols=["return", "volatility"])
    
    print("Running Backtest...")
    df_results = run_backtest(df_classified, return_col="return", label_col="regime_label")
    
    print("Plotting Results...")
    plot_results(df_results)

if __name__ == "__main__":
    main()
