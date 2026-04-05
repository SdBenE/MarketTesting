"""
DataLog: ML model prediction tracking
"""
import pandas as pd
import numpy as np
from markettesting.formatting import pull_csv, pull_yf
from markettesting.stock_models import StockModel
from markettesting.formatting import flatten_from_yf
from matplotlib import pyplot

class DataLog:
    def __init__(self, model, ticker_name, use_download=False, time_period='10y'):
        self.core_model = model
        self.ticker_name = ticker_name
        self.prediction_table = pd.DataFrame()

        if use_download:
            self.base_data = pull_csv(ticker=ticker_name)
        else:
            self.base_data = pull_yf(ticker=ticker_name, time_period=time_period)
            self.base_data = flatten_from_yf(self.base_data)
    
    def determine_prediction_set(self, days=90):

        #reduction to the number of days is necessary due to
        #extensive runtime with larger base datasets

        comp_data_table = np.empty((self.core_model.sequence_length + days, self.base_data.shape[1]))
        comp_data_table[:self.core_model.sequence_length] = self.base_data.iloc[-(self.core_model.sequence_length):]
        comp_data_table = pd.DataFrame(comp_data_table, columns=self.base_data.columns)

        for i in range(0, days):
            #Use last sequence_length units
            print(f"Datalog: Testing {self.ticker_name} on day {i + 1}")

            comp_data_table.iloc[self.core_model.sequence_length + i] = pd.DataFrame(
                self.core_model.get_prediction(
                comp_data_table.iloc[i: self.core_model.sequence_length + i]
                ),
                columns=comp_data_table.columns
            )
        
        self.prediction_table = comp_data_table.iloc[-days:]
        print(comp_data_table.head)
        print(self.prediction_table.head)