"""
Functions to download and load price data for pairs trading.
"""

import yfinance as yf
import pandas as pd
from typing import Tuple


def download_price_data(
    ticker1: str, ticker2: str, start_date: str, end_date: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download historical price data for two stocks.
    
    Args:
        ticker1: First stock ticker (e.g., 'KO')
        ticker2: Second stock ticker (e.g., 'PEP')
        start_date: Start date in format 'YYYY-MM-DD'
        end_date: End date in format 'YYYY-MM-DD'
    
    Returns:
        Tuple of (prices_ticker1, prices_ticker2) DataFrames
    """
    prices1 = yf.download(ticker1, start=start_date, end=end_date)
    prices2 = yf.download(ticker2, start=start_date, end=end_date)
    
    return prices1, prices2


def load_csv_data(filepath: str) -> pd.DataFrame:
    """
    Load price data from a CSV file.
    
    Args:
        filepath: Path to the CSV file
    
    Returns:
        DataFrame with price data
    """
    return pd.read_csv(filepath, index_col=0, parse_dates=True)
