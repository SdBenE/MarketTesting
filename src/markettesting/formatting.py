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
import time


def graphData(tickerName):
    table = np.genfromtxt(f'MarketTesting/src/markettesting/dataFolder/{tickerName}.csv', delimiter=',', dtype=None, encoding=None)
    print(table)
    xAxis = list(range(0, len(table[2:,0])))
    yAxis = table[2:,0]

    yAxis = yAxis.astype(np.float64)

    plt.plot(xAxis, yAxis)

    
    plt.show()

def fileFormation(periodYears=1, downLoad=False):
    dataList = pd.read_csv('MarketTesting/src/markettesting/tickers.csv')
    tempDataList = dataList
    print(dataList)
    symbols = dataList['Symbol']


    duration = f"{periodYears}y"

    # emptyTickers = []

    # if not os.path.exists("/MarketTesting/src/markettesting/dataFolder"):
    #     os.makedirs("/MarketTesting/src/markettesting/dataFolder", exist_ok=True)

    for ticker in symbols:
        if os.path.exists(f'/home/sdbene/tf-env/MarketTesting/src/markettesting/dataFolder/{ticker}.csv'):
            print(f'File {ticker}.csv already exists! Skipping...')
            continue
        
        print(f"      CURRENT TICKER {ticker}      ")
        
        start = time.perf_counter()
        
        myTicker = yf.download(ticker, period=duration)
        # myTicker.columns = [None] * len(myTicker.columns)

        print(f"Download completed in {time.perf_counter() - start} [s]")

        if myTicker.empty or len(myTicker) < 10:
            print(f"[[{ticker}]] : TOOOOOO SMALLLLLLLL BYE BYE")
            # emptyTickers.append(ticker)

            dropIndex = tempDataList[dataList['Symbol'] == ticker].index
            tempDataList = tempDataList.drop(dropIndex)
        elif downLoad:
            myTicker.to_csv(f'MarketTesting/src/markettesting/dataFolder/{ticker}.csv')

        # print(emptyTickers)
        print(f"Datalist Length: {len(tempDataList)}")
        # print(tempDataList)
    
    tempDataList.to_csv('/home/enjamin_lmore/tf-env/MarketTesting/src/markettesting/tickers.csv')
