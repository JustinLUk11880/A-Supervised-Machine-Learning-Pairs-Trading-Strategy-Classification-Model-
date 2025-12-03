"""
Backtesting logic for pairs trading strategy.
"""

import pandas as pd
import numpy as np


def backtest_strategy(
    prices: pd.DataFrame,
    signals: pd.Series,
    initial_capital: float = 100000,
    trading_cost: float = 0.001
) -> dict:
    """
    Backtest a trading strategy.
    
    Args:
        prices: DataFrame with price data (should have two columns for the pair)
        signals: Series with trading signals (1, -1, 0)
        initial_capital: Initial capital for trading
        trading_cost: Trading cost as a fraction (0.001 = 0.1%)
    
    Returns:
        Dictionary with backtest results and metrics
    """
    returns = prices.pct_change()
    strategy_returns = returns * signals.shift(1)
    
    # Account for trading costs
    strategy_returns = strategy_returns - trading_cost * (signals.diff().abs() / 2)
    
    cumulative_returns = (1 + strategy_returns).cumprod()
    portfolio_value = initial_capital * cumulative_returns
    
    return {
        'portfolio_value': portfolio_value,
        'returns': strategy_returns,
        'cumulative_returns': cumulative_returns
    }


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate the Sharpe ratio of returns.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
    
    Returns:
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / 252
    sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    return sharpe


def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
    """
    Calculate the maximum drawdown.
    
    Args:
        cumulative_returns: Series of cumulative returns
    
    Returns:
        Maximum drawdown as a fraction
    """
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    return max_drawdown


def calculate_win_rate(returns: pd.Series) -> float:
    """
    Calculate the win rate (percentage of positive returns).
    
    Args:
        returns: Series of returns
    
    Returns:
        Win rate as a fraction
    """
    return (returns > 0).sum() / len(returns)
