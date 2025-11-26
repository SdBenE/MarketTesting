import yfinance as yf
import csv
#from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os
import dataFinder
import tensorflow as tf
import formatting
import modelCreation
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import warnings

warnings.filterwarnings('ignore')

# dataFinder.createData('2021-01-01', '2025-01-01')
# myData = dataFinder.parseData('AAPL')

# formatting.graphData('ADP')
# formatting.graphData('AAPL')

# print("\n")

print(tf.__version__)

myModel = modelCreation.LTSMModel(epochs=50, units=100)
myModel.createModel(durationYears=4)
myModel.trainModel()


# print(f"NP DATASET: {myData}\n")

# print(f"TF DATASET: {dataset}")