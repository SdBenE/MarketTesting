import yfinance as yf
import csv
#from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os
import dataFinder
import numpy as np
import tensorflow as tf

def graphData(tickerName):
    table = np.genfromtxt(f'MarketTesting/src/markettesting/dataFolder/{tickerName}.csv', delimiter=',', dtype=None, encoding=None)
    xAxis = list(range(0, len(table[:,0])))
    yAxis = table[:,0]

    