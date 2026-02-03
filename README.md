### Statistical Arbitrage with Machine Learning (ETF Pairs)
## Overview

This project implements an end-to-end statistical arbitrage research pipeline for highly related ETF pairs (e.g. SPY–IVV, XLF–KBE).
The goal is to evaluate whether supervised machine learning can improve classical pairs trading by filtering low-quality mean-reversion signals.

Rather than optimizing for headline returns, the project emphasizes:

robust time-series methodology

regime awareness

empirical comparison between classical and ML-enhanced strategies

## Motivation

Classical pairs trading strategies assume persistent mean reversion, which often breaks across regimes.
This project reframes pairs trading as a supervised learning problem, where the model predicts whether mean reversion is likely to occur over a fixed horizon, while trade direction remains rule-based.

Key research questions:

Can ML improve risk-adjusted performance without increasing trade frequency?

Do regime diagnostics (e.g. persistence measures) add predictive value?

When does added model complexity stop helping?

## Methodology
1. Data

Daily adjusted close prices (2020–2026)

ETF pairs with strong structural linkage (e.g. SPY–IVV, XLF–KBE)

Prices aligned and cleaned for missing data

## 2. Feature Engineering

Hedge ratio estimation and spread construction

Rolling Z-score of the spread

Regime diagnostics:

Hurst exponent (mean-reversion vs persistence)

Stationarity indicators (ADF-based)

Volatility and spread dynamics

## 3. Labeling (Supervised Learning)

The problem is framed as binary classification:

Label = 1 if the absolute Z-score decreases over the next N days
Label = 0 otherwise

This separates:

direction (determined by Z-score sign)

confidence (learned by the model)

## 4. Models

Logistic regression with time-series–aware train/test split

ML used strictly as a trade filter, not a signal generator

## 5. Strategies Compared

Baseline: classical Z-score pairs trading

ML-filtered: baseline trades filtered by predicted probability

Continuous sizing (tested): evaluated but empirically rejected for tightly linked ETF pairs

## 6. Evaluation

Equity curves (normalized capital)

Sharpe ratio

Maximum drawdown

Trade count and exposure time

Comparative analysis across strategies and pairs

Key Findings

Supervised ML filtering consistently improved Sharpe ratios (≈25–60%) over classical baselines without increasing trade frequency.

Continuous position sizing reduced drawdowns but degraded Sharpe, indicating that alpha in ETF pairs is event-driven rather than continuous.

Strict cointegration tests were not necessary for profitable mean-reversion behavior; empirical validation proved more informative.

Simpler, well-justified models outperformed more complex alternatives.

## Project Structure
├── data_raw/           # Raw price data
├── data_processed/     # Engineered features, datasets, predictions
├── notebooks/
│   ├── 01_data_download.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_label_creation.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_backtest_and_evaluation.ipynb
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── labeling.py
│   ├── models.py
│   ├── backtest.py
│   └── utils.py
└── README.md

## Technologies Used

Python (NumPy, pandas, matplotlib)

scikit-learn

statsmodels

Time-series–aware validation

Git / GitHub

Notes on Scope

Results are unleveraged and transaction costs are not included

The project is intended as a research and learning exercise, not a production trading system

Emphasis is placed on interpretability, robustness, and negative-result documentation

## Future Extensions

Portfolio of multiple pairs for diversification

Intraday data and higher-frequency signals

Transaction cost modeling

Alternative labeling schemes (return-based, multi-class)

## Disclaimer

This project is for educational and research purposes only and does not constitute financial advice.
