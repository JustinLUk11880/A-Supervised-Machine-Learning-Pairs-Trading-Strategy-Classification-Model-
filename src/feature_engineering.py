"""
Feature engineering for the spread between two stocks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

def rolling_correlation(x, y, window):
    n = len(x)
    output = np.full(n, np.nan, dtype=float)
    
    for i in range(window - 1, n):
        x_window = x[i - window + 1:i + 1]
        y_window = y[i - window + 1:i + 1]
        
        x_mean = np.mean(x_window)
        y_mean = np.mean(y_window)
        
        dx = x_window - x_mean
        dy = y_window - y_mean
        
        covarience = np.dot(dx, dy)/(window - 1)
        std_x = np.std(x_window, ddof=1)
        std_y = np.std(y_window, ddof=1)
        if (std_x > 0 and std_y > 0):
            output[i] = covarience / (std_x * std_y)
        else:
            output[i] = np.nan
    
    return output

def overall_correlation(x, y):
    n = len(x)
    
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    dx = x - x_mean
    dy = y - y_mean

    covariance = np.dot(dx, dy)/(n - 1)
    std_x = np.std(x, ddof=1)
    std_y = np.std(y, ddof=1)
    if (std_x > 0 and std_y > 0):
        return covariance / (std_x * std_y)
    else:
        return np.nan

def hedge_ratio(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    covariance = np.dot(x - x_mean, y - y_mean) / (len(x) - 1)
    variance = ((x - x_mean) ** 2).sum()/(len(x) - 1)
    return covariance / variance

def calculate_spread(price1: pd.Series, price2: pd.Series) -> pd.Series:
    """
    Calculate the spread between two price series.
    
    Args:
        price1: First price series
        price2: Second price series
    
    Returns:
        Spread series (price1 - price2)
    """
    return price1 - price2


def calculate_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate the z-score of a series using rolling mean and std.
    
    Args:
        series: Input series
        window: Rolling window size
    
    Returns:
        Z-score series
    """
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    zscore = (series - rolling_mean) / rolling_std
    return zscore


def engineer_features(spread: pd.Series, window: int = 20) -> pd.DataFrame:
    """
    Engineer features from the spread.
    
    Args:
        spread: Spread series
        window: Rolling window size
    
    Returns:
        DataFrame with engineered features
    """
    features = pd.DataFrame(index=spread.index)
    
    features['spread'] = spread
    features['zscore'] = calculate_zscore(spread, window)
    features['rolling_mean'] = spread.rolling(window=window).mean()
    features['rolling_std'] = spread.rolling(window=window).std()
    features['rolling_min'] = spread.rolling(window=window).min()
    features['rolling_max'] = spread.rolling(window=window).max()
    
    return features.dropna()
