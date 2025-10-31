import yfinance as yf
import csv
#from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os

myTicker = 'NVDA'

myTestTicker = yf.Ticker(myTicker)

myData = yf.download(myTicker, period='max')

if not (os.path.exists(f'dataFolder/{myTicker}.csv')):
    myData.to_csv(f'dataFolder/{myTicker}.csv', index=False)

myData.plot()
plt.show()