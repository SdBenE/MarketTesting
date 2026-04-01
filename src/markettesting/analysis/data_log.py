"""
DataLog: ML model prediction tracking
"""
import pandas as pd
from markettesting.formatting import pull_csv, pull_yf
from markettesting.stock_models import StockModel
from markettesting.formatting import flatten_from_yf

class DataLog:
    def __init__(self, model, ticker_name, use_download=False, time_period='1y'):
        self.core_model = model
        self.ticker_name = ticker_name
        self.prediction_table = pd.dataFrame()

        if use_download:
            self.base_data = pull_csv(ticker=ticker_name)
        else:
            self.base_data = pull_yf(ticker=ticker_name, time_period=time_period)
            self.base_data = flatten_from_yf(self.base_data)
    
    def determine_prediction_set(self, days=90):
        comp_data_table = self.prediction_table

        for i in range(0, days):
            #Use last sequence_length units
            prediction_slice = self.core_model.get_prediction(
                comp_data_table[self.core_model.sequence_length:])