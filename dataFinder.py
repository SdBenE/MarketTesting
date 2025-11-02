import yfinance as yf
#from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os
#import tensorflow as tf
import numpy as np

def createData():
    myData = pd.read_csv('tickers.csv')

    dataList = myData['Symbol']
    for symbol in dataList: #The entire NASDAQ in the palm of my hand
            if not os.path.exists(f'{symbol}.csv'):
                myTicker = yf.download(symbol, start='2021-01-01', end='2025-06-01')
                myTicker.to_csv(f"dataFolder/{symbol}.csv", index=False)

def parseData(tickerName):
     return np.genfromtxt(f"dataFolder/{tickerName}.csv", delimiter=',', dtype=None, encoding=None)