"""
Machine learning models and training utilities for pairs trading.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def train_logistic_regression(X_train, y_train, **kwargs):
    """
    Train a logistic regression model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        **kwargs: Additional arguments for LogisticRegression
    
    Returns:
        Trained LogisticRegression model
    """
    model = LogisticRegression(**kwargs)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, **kwargs):
    """
    Train a random forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        **kwargs: Additional arguments for RandomForestClassifier
    
    Returns:
        Trained RandomForestClassifier model
    """
    model = RandomForestClassifier(**kwargs)
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train, **kwargs):
    """
    Train a gradient boosting classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        **kwargs: Additional arguments for GradientBoostingClassifier
    
    Returns:
        Trained GradientBoostingClassifier model
    """
    model = GradientBoostingClassifier(**kwargs)
    model.fit(X_train, y_train)
    return model


def scale_features(X_train, X_test=None):
    """
    Standardize features using StandardScaler.
    
    Args:
        X_train: Training features
        X_test: Test features (optional)
    
    Returns:
        Scaled X_train and X_test (if provided)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    
    return X_train_scaled
