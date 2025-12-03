"""
Create supervised labels for pairs trading signals.
"""

import pandas as pd
import numpy as np


def create_labels(
    spread: pd.Series,
    zscore: pd.Series,
    long_threshold: float = -1.0,
    short_threshold: float = 1.0
) -> pd.Series:
    """
    Create trading labels based on z-score thresholds.
    
    Args:
        spread: Spread series
        zscore: Z-score of the spread
        long_threshold: Z-score threshold for long signals (typically < 0)
        short_threshold: Z-score threshold for short signals (typically > 0)
    
    Returns:
        Series with labels: 1 (long), -1 (short), 0 (no trade)
    """
    labels = pd.Series(0, index=spread.index, dtype=int)
    
    # Long signal: spread is low (mean reversion)
    labels[zscore < long_threshold] = 1
    
    # Short signal: spread is high (mean reversion)
    labels[zscore > short_threshold] = -1
    
    return labels


def create_returns_labels(
    spread: pd.Series,
    periods_ahead: int = 5,
    threshold: float = 0.01
) -> pd.Series:
    """
    Create labels based on future returns (forward-looking).
    
    Args:
        spread: Spread series
        periods_ahead: Number of periods to look ahead
        threshold: Return threshold for classification
    
    Returns:
        Series with labels: 1 (positive return), -1 (negative return), 0 (neutral)
    """
    future_returns = spread.shift(-periods_ahead) / spread - 1
    
    labels = pd.Series(0, index=spread.index, dtype=int)
    labels[future_returns > threshold] = 1
    labels[future_returns < -threshold] = -1
    
    return labels
