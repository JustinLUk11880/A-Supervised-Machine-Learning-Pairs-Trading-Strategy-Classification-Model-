"""
Utility functions for plotting, metrics, and general helpers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve


def plot_prices_and_spread(price1, price2, spread, title="Prices and Spread"):
    """
    Plot two price series and their spread.
    
    Args:
        price1: First price series
        price2: Second price series
        spread: Spread series
        title: Title for the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    
    ax1.plot(price1.index, price1, label='Price 1', alpha=0.7)
    ax1.plot(price2.index, price2, label='Price 2', alpha=0.7)
    ax1.set_title(f"{title} - Prices")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(spread.index, spread, label='Spread', color='purple', alpha=0.7)
    ax2.set_title(f"{title} - Spread")
    ax2.set_ylabel("Spread")
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_zscore(spread, zscore, threshold=2, title="Z-Score"):
    """
    Plot spread and its z-score with trading thresholds.
    
    Args:
        spread: Spread series
        zscore: Z-score series
        threshold: Threshold for trading signals
        title: Title for the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    
    ax1.plot(spread.index, spread, label='Spread', color='blue', alpha=0.7)
    ax1.set_title(f"{title} - Spread")
    ax1.set_ylabel("Spread")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(zscore.index, zscore, label='Z-Score', color='orange', alpha=0.7)
    ax2.axhline(y=threshold, color='r', linestyle='--', label=f'Upper Threshold ({threshold})')
    ax2.axhline(y=-threshold, color='g', linestyle='--', label=f'Lower Threshold (-{threshold})')
    ax2.set_title(f"{title} - Z-Score")
    ax2.set_ylabel("Z-Score")
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_portfolio_performance(portfolio_value, title="Portfolio Performance"):
    """
    Plot portfolio value over time.
    
    Args:
        portfolio_value: Series of portfolio values
        title: Title for the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(portfolio_value.index, portfolio_value, linewidth=2, color='green')
    ax.set_title(title)
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    
    return fig


def print_model_evaluation(y_true, y_pred, y_pred_proba=None):
    """
    Print classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
    """
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    if y_pred_proba is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1], multi_class='ovr')
            print(f"\nROC AUC Score: {roc_auc:.4f}")
        except:
            pass
