"""
Functions to download and load price data for pairs trading.
"""

import yfinance as yf
import pandas as pd
from typing import Tuple


def download_price_data(
    tickets: list, start_date: str, end_date: str
) -> pd.DataFrame:
    price_data = yf.download(tickets, 
                            start=start_date, 
                            end=end_date, 
                            interval="1d",
                            auto_adjust=True)

    price_data.head()
    return price_data


def load_csv_data(filepath: str, ticker1: str, ticker2: str):
    
    prices = pd.read_csv(filepath, index_col=0, parse_dates=True).dropna()
    price1 = prices[ticker1].values
    price2 = prices[ticker2].values

    return prices, price1, price2
