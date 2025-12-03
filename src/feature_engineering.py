"""
Feature engineering for the spread between two stocks.
"""

import pandas as pd
import numpy as np
from scipy import stats


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
