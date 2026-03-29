"""
DataLog: ML model prediction tracking
"""
import pandas as pd
from markettesting.formatting import pull_csv, pull_yf

class DataLog:
    def __init__(self, model, ticker_name, use_download=False, time_period='1y'):
        self.core_model = model
        self.ticker_name = ticker_name
        self.prediction_table = pd.dataFrame()

        if use_download:
            self.base_data = pull_csv(ticker=ticker_name)
        else:
            self.base_data = pull_yf(ticker=ticker_name, time_period=time_period)
    
