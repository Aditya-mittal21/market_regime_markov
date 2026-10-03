# Market Regime Detection with Hidden Markov Models

This repository contains a quantitative trading strategy that employs a Hidden Markov Model (HMM) to classify market regimes and optimize trading decisions.

## Overview

The strategy analyzes the SPDR S&P 500 ETF Trust (SPY) by engineering features such as daily returns and rolling volatility. A Gaussian Hidden Markov Model is then trained to classify the current market environment into hidden states (e.g., "bull" or "bear" regimes). The trading strategy simulates taking a long position only during identified "bull" regimes, minimizing exposure during high-volatility "bear" market periods.

## Features

- **Data Fetching:** Automatically downloads historical price data via Yahoo Finance.
- **Feature Engineering:** Calculates rolling volatility and daily returns to feed the HMM.
- **HMM Classification:** Utilizes `hmmlearn` to train a Gaussian HMM and label market regimes.
- **Backtesting & Evaluation:** Simulates the trading strategy, compares it against a buy-and-hold approach, and calculates the Sharpe ratio.

## Installation

1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main pipeline to fetch data, train the model, and view the backtest results:

```bash
python main.py
```

## Results

The backtest was conducted on real historical data for the SPY ETF from **January 1, 2020** to **August 1, 2025**. 

By filtering out high-volatility, downward-trending market regimes, the strategy significantly improved risk-adjusted returns compared to a traditional buy-and-hold portfolio:
- **Market Sharpe Ratio:** `0.753`
- **Strategy Sharpe Ratio:** `1.123`
- **Market Final Cumulative Return:** `2.13x`
- **Strategy Final Cumulative Return:** `2.22x`


