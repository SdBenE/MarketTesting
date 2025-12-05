import yfinance as yf
import csv
#from pathlib import Path
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import dataFinder
import numpy as np
import tensorflow as tf

matplotlib.use('TkAgg')

def graphData(tickerName):
    table = np.genfromtxt(f'MarketTesting/src/markettesting/dataFolder/{tickerName}.csv', delimiter=',', dtype=None, encoding=None)
    print(table)
    xAxis = list(range(0, len(table[2:,0])))
    yAxis = table[2:,0]

    yAxis = yAxis.astype(np.float64)

    plt.plot(xAxis, yAxis)

    
    plt.show()

def fileFormation(periodYears=1):
    dataList = pd.read_csv('MarketTesting/src/markettesting/tickers.csv')
    tempDataList = dataList
    print(dataList)
    symbols = dataList['Symbol']


    duration = f"{periodYears}y"

    # emptyTickers = []

    for ticker in symbols:
        myTicker = yf.download(ticker, period=duration)
        # myTicker.columns = [None] * len(myTicker.columns)

        print(f"      CURRENT TICKER {ticker}")

        if myTicker.empty or len(myTicker) < 10:
            print(f"[[{ticker}]] : TOOOOOO SMALLLLLLLL BYE BYE")
            # emptyTickers.append(ticker)

            dropIndex = tempDataList[dataList['Symbol'] == ticker].index
            tempDataList = tempDataList.drop(dropIndex)

        # print(emptyTickers)
        print(f"Datalist Length: {len(tempDataList)}")
        # print(tempDataList)
    