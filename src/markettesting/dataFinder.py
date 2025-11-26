import yfinance as yf
#from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np

def createData(startDate, endDate):
    myData = pd.read_csv('MarketTesting/src/markettesting/tickers.csv')

    

    dataList = myData['Symbol']
    for symbol in dataList: #The entire NASDAQ in the palm of my hand
            if not os.path.exists(f'{symbol}.csv'):
                myTicker = yf.download(symbol, start=startDate, end=endDate)
                print(myTicker.head())
                myTicker.to_csv(f"MarketTesting/src/markettesting/dataFolder/{symbol}.csv", index=True)

def parseData(tickerName):
    array = np.genfromtxt(f"MarketTesting/src/markettesting/dataFolder/{tickerName}.csv", delimiter=',', dtype=None, encoding=None)
    array = array[2:]
    # print(array)
    return array
