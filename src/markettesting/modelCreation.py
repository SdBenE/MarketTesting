import yfinance as yf
import csv
#from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import dataFinder
import tensorflow as tf
import formatting
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

class LTSMModel:
    def __init__(self, epochs=25, units=50):
        self.model = Sequential()
        self.epochs = epochs
        self.units = units
    
    def createModel(self, durationYears=1, sequenceLength=60):
        self.timePeriod = f"{durationYears}y"
        self.sequenceLength = sequenceLength

        #Layer 1
        self.model.add(LSTM(units=self.units, return_sequences=True, input_shape=(durationYears * 365, 1)))
        self.model.add(Dropout(0.2))

        #Layer 2
        self.model.add(LSTM(units=self.units, return_sequences=True))
        self.model.add(Dropout(0.2))

        self.model.add(LSTM(units=self.units, return_sequences=False))

        self.model.add(Dense(units=1))

        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def dataSequence(self, trainingData):
        x = []
        y = []

        for i in range(self.sequenceLength, len(trainingData)):
            x.append(trainingData[i - self.sequenceLength:i, 0])
            y.append(trainingData[i, 0])

        # print(f"X Array {np.array(x)}")
        # print(f"Y Array {np.array(y)}")

        return np.array(x), np.array(y)

    def trainModel(self):
        dataList = pd.read_csv('MarketTesting/src/markettesting/tickers.csv')
        dataList = dataList['Symbol']

        for ticker in dataList:
            rawData = yf.download(ticker, period=self.timePeriod)

            rawData.columns = rawData.columns.droplevel(level=1)
            rawData = rawData[['Close', 'High', 'Low', 'Open', 'Volume']]
            rawData.index.name = 'Date'
            rawData.index = pd.to_datetime(rawData.index)

            rawData['Close'] = pd.to_numeric(rawData['Close'], errors='coerce')
            rawData.dropna(subset=['Close'], inplace=True)

            # print(rawData.head())

            # rawData['Date'] = pd.to_datetime(rawData['Date'])
            # rawData.set_index('Date', inplace=True)#FIXME: Error with "Date" column
            MIN_DATA_POINTS = self.sequenceLength + 10

            if len(rawData) == 0:
                print(f"{ticker} is empty! Skipping...")
                continue
            
            rawData.dropna(subset=['Close'], inplace=True)

            if len(rawData) < MIN_DATA_POINTS:
                print(f"{ticker} IS EMPTY: SKIPPING...")
                continue
            else:
                usedData = rawData['Close'].values

                indexCutoff = int(len(usedData) * 0.7)
                trainingPortion = usedData[:indexCutoff]
                testingPortion = usedData[indexCutoff:]

                scale = MinMaxScaler(feature_range=(0,1))

                scale.fit(trainingPortion.reshape(-1,1))

                trainData = scale.transform(trainingPortion.reshape(-1,1))
                testData = scale.transform(testingPortion.reshape(-1,1))

                print(f"Reshaping {ticker}")

                xVals, yVals = self.dataSequence(trainData)
                x3D = np.reshape(xVals, (xVals.shape[0], xVals.shape[1], 1))

                # print(x3D)

                self.model.fit(x=x3D, y=yVals, epochs=self.epochs, batch_size=32)


                testSeq, testTarget = self.dataSequence(testData)

                unscaledTargets = usedData[indexCutoff:]

                scaledPredictions = self.model.predict(testSeq, verbose=0)
                fixedPredicts = scale.inverse_transform(scaledPredictions)

                actualTargets = unscaledTargets.reshape(-1, 1)

                actualFromScaled = scale.inverse_transform(testTarget.reshape(-1,1))

                mae = mean_absolute_error(actualFromScaled, fixedPredicts)
                print(f"MEAN ABSOLUTE ERROR FROM {ticker}: ${mae}")

                self.model.save('/home/enjamin_lmore/tf-env/MarketTesting/src/markettesting/myModel.keras')



