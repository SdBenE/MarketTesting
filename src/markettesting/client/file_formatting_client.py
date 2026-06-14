"""
Base client code for file downloading
"""

import warnings
import pandas as pd
from markettesting.formatting import file_formation, volatility_file_formation, flatten_from_yf, pull_data_csv
from markettesting.config import EXTERNAL_FACTORS_DIR, DATA_FOLDER_DIR

warnings.filterwarnings('ignore')

volatility_file_formation(period_years='max', download=True)

stock_data = pull_data_csv('ticker', 'AAPL')
print(stock_data.head())
stock_data = flatten_from_yf(stock_data)
print(stock_data.head())

vol_data = pull_data_csv('volatility')
print(vol_data.head())
vol_data = flatten_from_yf(vol_data)
print(vol_data.head())

# file_formation(period_years=10, download=True)