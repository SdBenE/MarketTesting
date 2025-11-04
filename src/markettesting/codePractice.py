import yfinance as yf
import csv
#from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os
import dataFinder
import tensorflow as tf
import formatting

#dataFinder.createData()
myData = dataFinder.parseData('AAPL')

dataset = tf.data.Dataset.from_tensor_slices(myData)

print(myData)