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

def hurst_exponent(ts, max_lag=100):
    """
    Estimate the Hurst exponent of a time series.
    ts: array-like (spread)
    """
    ts = np.asarray(ts, dtype=float)
    ts = ts[np.isfinite(ts)]

    # lag is the time interval to analyze
    lags = range(2, max_lag)
    tau = [] # store the standard deviations of the differences in lags

    for lag in lags:
        diff = ts[lag:] - ts[:-lag] # difference with lag
        tau.append(np.sqrt(np.var(diff))) # store in tau

    # tau(L) ∝ L^H
    # log(tau) = H × log(L) + constant
    # log-log regression
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    H = poly[0]

    return H