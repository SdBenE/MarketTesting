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
import frontend
#import dataLog
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import warnings

warnings.filterwarnings('ignore')

frontend.intro()

# entry = input(f'<Entry> : ')

# while entry != 'exit' or entry != 'q' or entry != 'quit':
#     frontend.checkInput(entry)
#     entry = input(f'<Entry> : ')

# myDataLog = dataLog.DataLog()
# myDataLog.importDataLog()
# myDataLog.addElement('NVDA', '30d')

# dataFinder.createData('2021-01-01', '2025-01-01')
# myData = dataFinder.parseData('AAPL')

# formatting.graphData('ADP')
# formatting.graphData('AAPL')

# print("\n")


# formatting.fileFormation(4)

myModel = modelCreation.LTSMModel(epochs=50, units=300)
myModel.createModel(durationYears=8)
myModel.trainModel()

# print(f"NP DATASET: {myData}\n")

# print(f"TF DATASET: {dataset}")
