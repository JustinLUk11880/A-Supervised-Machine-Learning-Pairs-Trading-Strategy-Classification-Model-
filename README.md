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
