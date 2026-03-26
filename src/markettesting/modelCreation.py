import yfinance as yf
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import dataFinder
import tensorflow as tf
import formatting
from datetime import date
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.callbacks import EarlyStopping

class LTSMModel:
    def __init__(self, epochs=25, units=50, name="StockModel"):
        self.name = name
        self.model = Sequential()
        self.epochs = epochs
        self.units = units
        self.predictionTable = None
        self.scaler = None

    def importModel(self):
        self.model = load_model(f"{self.name}.keras")

    def createModel(self, durationYears=1, sequenceLength=100, numFeatures=5):
        self.timePeriod = f"{durationYears}y"
        self.sequenceLength = sequenceLength

        #Layer 1
        self.model.add(LSTM(units=self.units, return_sequences=True, input_shape=(self.sequenceLength, numFeatures)))
        self.model.add(Dropout(0.2))

        #Layer 2
        self.model.add(LSTM(units=self.units, return_sequences=True))
        self.model.add(Dropout(0.2))

        #Layer 3
        self.model.add(LSTM(units=self.units, return_sequences=False))

        self.model.add(Dense(units=1))

        self.earlyStopSystem = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def getPrediction(self, inputData):
        #TODO: Complete Prediction method
        #TODO: Identify dense layer changes with numFeatures
        if len(inputData) != self.sequenceLength:
            raise ValueError(f"getPrediction: Input data shape must match model's sequenceLength in:{len(inputData)} model:{self.sequenceLength}")
        
    def dataSequence(self, trainingData):
        xList = []
        yList = []
        
        i = 0

        for j in range(self.sequenceLength, len(trainingData)):
            xList.append(trainingData.iloc[i:j])

            yList.append(trainingData.at[j, 'Close']) #Adds the following day to it
            i+=1

        return np.array(xList), np.array(yList)

    def pullCSV(self, ticker, mainDir='MarketTesting/src/markettesting/dataFolder/'):
        if os.path.exists(f'{mainDir}{ticker}.csv'):
            dataSet = pd.read_csv(f'{mainDir}{ticker}.csv')
        else:
            return None

        dataSet = dataSet.iloc[2:].copy()
        dataSet.columns = dataSet.columns.get_level_values(0)
        dataSet = dataSet.reset_index(drop=True)

        dataSet = dataSet.drop(labels='Price', axis=1)
        dataSet.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
        return dataSet
        
    def pullYF(self, ticker):
        dataSet = yf.download(ticker, period=self.timePeriod)
        dataSet = dataSet.reset_index(drop=True) 

        if len(dataSet) < 10:
            return None

        dataSet.columns = dataSet.columns.get_level_values(0)
        dataSet.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
        return dataSet

    def trainModel(self, useDownload=True, useOldScaler=False):
        dataList = pd.read_csv('MarketTesting/src/markettesting/tickers.csv')
        dataList = dataList['Symbol']

        for ticker in dataList:
            print(f'     Current Ticker: {ticker}')

            if useDownload:
                rawData = self.pullCSV(ticker)
            else:
                rawData = self.pullYF(ticker)

            if rawData is None or len(rawData) < 100:
                print(f"Ticker {ticker} is too small or doesn't exist")
                print(f"Skipping {ticker}...")
                continue

            #SCALING
            # scaler = MinMaxScaler(feature_range=(0,1))
            scaler = StandardScaler()

            #This uses global scaling based on the whole dataset to check proper values

            # scaledData = self.valueScaler.transform(rawData)
            scaledData = scaler.fit_transform(rawData)
            scaledData = pd.DataFrame(scaledData, columns=rawData.columns)

            #TODO:Remove before use
            print(scaledData.describe())

            #ORGANIZING
            xFull, yFull = self.dataSequence(scaledData)

            #SPLITTING DATA
            splitIndex = int(0.8 * len(yFull)) #Integer casting for proper index

            xTrain = xFull[:splitIndex]
            yTrain = yFull[:splitIndex]
            xTest = xFull[splitIndex+1:]
            yTest = yFull[splitIndex+1:]

            history = self.model.fit(
                xTrain, 
                yTrain, 
                epochs=self.epochs, 
                batch_size=256, 
                callbacks=[self.earlyStopSystem],
                validation_data=(xTest, yTest)
            )

            print(f'CLASS SIZE: {sys.getsizeof(self)} bytes')
            print(f'MODEL SIZE: {sys.getsizeof(self.model)} bytes')

        self.model.save(f'MarketTesting/src/markettesting/{self.name}.keras')

    # def getPrediction(self):
