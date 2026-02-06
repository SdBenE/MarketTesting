import os
import pandas as pd
import dataFinder

class DataLog:
    def __init__(self):
        self.dataLog = pd.DataFrame
        

    def importDataLog(self):
        if os.path.exists('dataLog.csv'):
            self.dataLog = pd.read_csv('dataLog.csv')
        else:
            self.dataLog = pd.DataFrame({'Symbol':[None], 'Initial Value': [None], 'Duration': [None], 'Current': [None], 'Margin': [None]}) #TODO: Complete line for initial values in dataframe
            print(self.dataLog.head())
            self.dataLog.to_csv('dataLog.csv')

    def addElement(self, symbol, duration):
        newElement = pd.DataFrame({'Symbol': [symbol], 'Initial Value': [dataFinder.getValue(symbol)], 'Duration': [duration],
                                   'Current': [None], 'Margin': [None]})
        self.dataLog = pd.concat([self.dataLog, newElement], ignore_index=True)
        print(self.dataLog.head())
