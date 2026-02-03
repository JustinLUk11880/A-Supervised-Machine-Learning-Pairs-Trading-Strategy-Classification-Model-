import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Annualized Sharpe ratio (no risk-free rate).
    """
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return np.nan
    return np.sqrt(periods_per_year) * (r.mean() / r.std())


def max_drawdown(equity: pd.Series) -> float:
    """
    Maximum drawdown of an equity curve.
    """
    eq = equity.dropna()
    if eq.empty:
        return np.nan
    running_max = eq.cummax()
    drawdown = (eq / running_max) - 1.0
    return float(drawdown.min())


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DataFrame index is a sorted DatetimeIndex.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def trade_count(position: pd.Series) -> int:
    """
    Counts number of position changes (proxy for trades).
    """
    return int((position.diff().fillna(0) != 0).sum())
