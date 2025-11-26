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
    