"""
STOCK MODEL CLASS
"""
import os
import yfinance as yf
import numpy as np
import pandas as pd
import pickle
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import RobustScaler
from markettesting.config import BASE_DIRECTORY, DATA_FOLDER_DIR, TICKER_DIR
from markettesting.formatting import pull_ticker_csv, pull_yf, check_invalid_ticker
from markettesting.stock_models.scalers import create_ticker_scaler

class StockModel:
    """
    DEFAULT STOCK MODEL FOR MARKETTESTING
    """
    def __init__(self, sequence_length=100, num_features=5, time_period ='4y', model_import=False):
        """
        Default constructor for StockModel
        """
        self.time_period = time_period
        self.model = Sequential()
        self.epochs = 25
        self.units = 100
        self.num_features = num_features
        self.sequence_length = sequence_length
        self.scaler = None

        if model_import:
            self.import_model()
            self.import_pickle_scaler()
            print("Importing model from StockModel.keras")
        else:
            self.create_model()

        self.early_stop_system = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            min_delta=0.0001
        )

    def import_model(self, model_directory="StockModel.keras"):
        """Imports model in .keras format"""
        self.model = load_model(BASE_DIRECTORY / model_directory)

    def import_pickle_scaler(self, pickle_directory="StockModel.pkl"):
        self.scaler = pickle.load(open(BASE_DIRECTORY / pickle_directory, "rb"))

    def create_model(self):
        """
        Creates default LSTM model for StockModel Class
        Stored in self.model
        """
        #Layer 1
        self.model.add(LSTM(units=self.units,
                            return_sequences=True,
                            input_shape=(self.sequence_length, self.num_features)))
        self.model.add(Dropout(0.2))

        #Layer 2
        self.model.add(LSTM(units=self.units, return_sequences=False))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(self.num_features))

        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def get_prediction(self, input_data, return_close=False):
        """Getter for predictions"""

        #SCALE AND RESHAPE TO TENSOR
        scaled_input = self.scaler.transform(input_data)
        scaled_input = np.expand_dims(scaled_input, 0)

        #PREDICT
        scaled_prediction = self.model.predict(scaled_input)
        if return_close:
            unscaled_prediction = self.scaler.inverse_transform(scaled_prediction)
            return unscaled_prediction[0,0]
        else:
            return self.scaler.inverse_transform(scaled_prediction)
    
    def data_sequence(self, training_data):
        """
        Sequences time-series data windows for training
        Bases the size of the windows on stored sequence length field
        """
        x_list = []
        y_list = []
        i = 0

        for j in range(self.sequence_length, len(training_data)):
            x_list.append(training_data.iloc[i:j])

            y_list.append(training_data.iloc[j]) #TODO: Remove if fails
            i+=1

        return np.array(x_list), np.array(y_list)

    def train_model_single_set(self, ticker, raw_data, use_single_download=True, save_dir="StockModel", batch_size=128):
        """
        Single-ticker model preprocessing and data push-through
        """
        if raw_data is None:
            print(f"{ticker} not found!")
            print("Skipping...")
            #Exit training w/ this ticker if conditions not met
            return None

        raw_data = raw_data.apply(pd.to_numeric, errors='coerce').dropna()

        if check_invalid_ticker(ticker, raw_data):
            print("Skipping...")
            #Exit training w/ this ticker if conditions not met
            return

        #SCALING
        scaled_data = self.scaler.transform(raw_data)
        scaled_data = pd.DataFrame(scaled_data, columns=raw_data.columns)

        #WINDOWING
        x_full, y_full = self.data_sequence(scaled_data)

        #SPLITTING DATA
        split_index = int(0.8 * len(y_full)) #Integer casting for proper index

        x_train = x_full[:split_index]
        y_train = y_full[:split_index]
        x_test = x_full[split_index+1:]
        y_test = y_full[split_index+1:]

        print(f"X training range: {x_train.min()} to {x_train.max()}")
        print(f"Y training range: {y_train.min()} to {y_train.max()}")

        if (np.abs(x_train).max() > 10):
            print(f"Scaled data for {ticker} is too large!!")
            return

        self.model.fit(
            x_train,
            y_train,
            epochs=self.epochs,
            batch_size=batch_size,
            callbacks=[self.early_stop_system],
            validation_data=(x_test, y_test)
        )

        self.model.save(BASE_DIRECTORY / f'{save_dir}.keras')

    def train_model(self, use_download=True, save_dir="StockModel", batch_size=128):
        """
        StockModel preprocessing and data push-through
        """
        if self.scaler == None:
            self.scaler = create_ticker_scaler(time_period=self.time_period, use_download=use_download)

        data_list = pd.read_csv(TICKER_DIR)
        data_list = data_list['Symbol']

        for ticker in data_list:
            print(f'     Current Ticker: {ticker}')

            if use_download:
                raw_data = pull_ticker_csv(ticker)
            else:
                raw_data = pull_yf(ticker, time_period=self.time_period)

            self.train_model_single_set(ticker,
                                        raw_data,
                                        use_single_download=use_download,
                                        save_dir=save_dir,
                                        batch_size=batch_size
            )
