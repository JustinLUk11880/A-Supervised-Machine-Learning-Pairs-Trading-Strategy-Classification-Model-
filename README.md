# A-Supervised-Machine-Learning-Pairs-Trading-Strategy-Classification-Model-

# Supervised Machine Learning Pairs Trading Strategy

This project implements a **supervised machine learning pairs trading strategy**.

We:

- Select a **pair of correlated stocks** (e.g. KO & PEP or AAPL & MSFT)
- Build a **spread** between them
- Engineer features from the spread (z-score, rolling stats, etc.)
- Create **labels** for mean-reversion trades (long / short / no trade)
- Train a **supervised ML model** (e.g. Logistic Regression)
- Use model predictions to generate **trading signals**
- **Backtest** the strategy and evaluate performance (Sharpe ratio, drawdown, etc.)

The project is written in **Python**, uses a **`venv` virtual environment**, and is developed in **VS Code** with **Jupyter notebooks**.

---

## 🔧 Tech Stack

- Python 3.10+
- `venv` (built-in virtual environment)
- VS Code + Jupyter extension
- Main libraries:
  - `pandas`, `numpy`, `scipy`
  - `scikit-learn`
  - `matplotlib`, `seaborn`
  - `yfinance` (price data)

---

## 📁 Project Structure

Proposed folder layout:

```text
.
├── data_raw/                # Raw downloaded price data
├── data_processed/          # Cleaned data / features / labels
│
├── notebooks/
│   ├── 01_data_download.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_label_creation.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_backtest_and_evaluation.ipynb
│
├── src/
│   ├── data_loader.py           # Functions to download & load data
│   ├── feature_engineering.py   # Spread & feature calculations
│   ├── labeling.py              # Create supervised labels for trades
│   ├── models.py                # ML models and training helpers
│   ├── backtest.py              # Backtesting logic
│   └── utils.py                 # Helper utilities (plotting, metrics, etc.)
│
├── requirements.txt
└── README.md
```

01_data_download.ipynb
Goal

Download raw prices for the two tickers and save them so every later notebook starts from the same file.

What to implement

Set tickers + date range

Download adjusted close prices

Clean (drop NaNs)

Save to data_raw/

Done criteria

You have a file like: data_raw/prices_KO_PEP.csv

Reloading it gives a DataFrame with Date index and 2 columns.

Save format recommendation: one clean CSV with just two close series.

02_feature_engineering.ipynb
Goal

Load raw data, compute hedge ratio, spread, rolling mean/std, z-score, and save features.

What to implement

Load data*raw/prices*\*.csv

Compute hedge ratio β (OLS formula or statsmodels)

Compute spread = price2 - β\*price1

Compute rolling mean/std and zscore

Save to data_processed/:

spread.csv

features.csv (contains spread, zscore, rolling stats)

Done criteria

Plots render:

spread with mean

z-score with ±2 lines

You have data_processed/features.csv with aligned dates and no weird length errors.

- **Hurst exponent analysis completed:** Hurst exponent computed for the spread (mean-reversion indicated if H < 0.5).

03_label_creation.ipynb
Goal

Create supervised learning labels y from your features and save a single modeling dataset.

What to implement

Pick ONE labeling approach for the skeleton. Start simple:

Recommended label (binary, easiest)

“Will |z| decrease over the next N days?”

𝑦
𝑡
=
1
if
∣
𝑧
𝑡

- 𝑁
  ∣
  <
  ∣
  𝑧
  𝑡
  ∣
  ,
  else
  0
  y
  t
  ​

=1 if ∣z
t+N
​

∣<∣z
t
​

∣, else 0

Steps:

Load data_processed/features.csv

Choose horizon N (5 or 10 trading days)

Create label y

Combine X features + y into one dataset

Save to data_processed/dataset.csv

Done criteria

dataset.csv exists with columns:

feature columns (zscore, spread, vol, etc.)

y

y has both classes (not all 0s or all 1s)

04_model_training.ipynb
Goal

Train a supervised model using time-based split and save predictions for backtesting.

What to implement

Load data_processed/dataset.csv

Time-based split (70/30 or by date)

Train baseline model:

Logistic Regression (best starter)

Evaluate basic ML metrics:

accuracy

precision/recall

confusion matrix

Save predictions:

predicted probability proba

predicted class y_hat
to data_processed/predictions.csv

Done criteria

Model trains without errors

You have predictions.csv aligned by date

You can see metrics printed

05_backtest_and_evaluation.ipynb
Goal

Backtest:

baseline z-score strategy

ML-filtered strategy
and compare performance.

What to implement
A) Baseline strategy (rule-based)

Enter long if z < -2

Enter short if z > +2

Exit when z crosses 0 (or |z| < 0.5)

B) ML strategy (simple filter)

Only trade when:

z-score is extreme AND

ML probability of reversion is high (e.g., proba > 0.55)

Returns

Compute:

spread return ≈ ret2 - beta\*ret1

strategy return = position(t-1) \* spread_ret(t)

Evaluation

equity curve

Sharpe ratio

max drawdown

number of trades

baseline vs ML plot

Done criteria

Two equity curves plotted (baseline vs ML)

Printed Sharpe + max drawdown + trade count

You can explain which is better and why
