from config import BASE_DIRECTORY, DATA_FOLDER_DIR, TICKER_DIR
import yfinance as yf
import numpy as np
import pandas as pd
import os
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping

class StockModel:
    def __init__(self, epochs=25, units=50, name="StockModel"):
        self.name = name
        self.model = Sequential()
        self.epochs = epochs
        self.units = units
        self.prediction_table = None
        self.scaler = None

    def import_model(self):
        self.model = load_model(BASE_DIRECTORY / f"{self.name}.keras")

    def create_model(self, duration_years=1, sequence_length=100, num_features=5):
        self.time_period = f"{duration_years}y"
        self.sequence_length = sequence_length

        #Layer 1
        self.model.add(LSTM(units=self.units,
                            return_sequences=True,
                            input_shape=(self.sequence_length, num_features)))
        self.model.add(Dropout(0.2))

        #Layer 2
        self.model.add(LSTM(units=self.units, return_sequences=True))
        self.model.add(Dropout(0.2))

        #Layer 3
        self.model.add(LSTM(units=self.units, return_sequences=False))

        self.model.add(Dense(units=1))

        self.early_stop_system = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def get_prediction(self, input_data):
        if len(input_data) != self.sequence_length:
            raise ValueError(f"get_prediction: Incorrect sequence_length")
    
    def data_sequence(self, training_data):
        x_list = []
        y_list = []
        i = 0

        for j in range(self.sequence_length, len(training_data)):
            x_list.append(training_data.iloc[i:j])

            y_list.append(training_data.at[j, 'Close']) #Adds the following day to it
            i+=1

        return np.array(x_list), np.array(y_list)

    def pull_csv(self, ticker, main_dir=DATA_FOLDER_DIR):
        if os.path.exists(DATA_FOLDER_DIR / f"{ticker}.csv"):
            data_set = pd.read_csv(DATA_FOLDER_DIR / f'{ticker}.csv')
        else:
            return None

        data_set = data_set.iloc[2:].copy()
        data_set.columns = data_set.columns.get_level_values(0)
        data_set = data_set.reset_index(drop=True)

        data_set = data_set.drop(labels='Price', axis=1)
        data_set.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
        return data_set
    
    def pull_yf(self, ticker):
        data_set = yf.download(ticker, period=self.time_period)
        data_set = data_set.reset_index(drop=True)

        if len(data_set) < 10:
            return None

        data_set.columns = data_set.columns.get_level_values(0)
        data_set.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
        return data_set

    def train_model(self, use_download=True):
        data_list = pd.read_csv(TICKER_DIR)
        data_list = data_list['Symbol']

        for ticker in data_list:
            print(f'     Current Ticker: {ticker}')

            if use_download:
                raw_data = self.pull_csv(ticker)
            else:
                raw_data = self.pull_yf(ticker)

            if raw_data is None or len(raw_data) < 100:
                print(f"Ticker {ticker} is too small or doesn't exist")
                print(f"Skipping {ticker}...")
                continue

            #SCALING
            # scaler = MinMaxScaler(feature_range=(0,1))
            scaler = StandardScaler()

            #This uses global scaling based on the whole data_set to check proper values

            # scaled_data = self.valueScaler.transform(raw_data)
            scaled_data = scaler.fit_transform(raw_data)
            scaled_data = pd.DataFrame(scaled_data, columns=raw_data.columns)

            #ORGANIZING
            x_full, y_full = self.data_sequence(scaled_data)

            #SPLITTING DATA
            split_index = int(0.8 * len(y_full)) #Integer casting for proper index

            x_train = x_full[:split_index]
            y_train = y_full[:split_index]
            x_test = x_full[split_index+1:]
            y_test = y_full[split_index+1:]

            self.model.fit(
                x_train,
                y_train,
                epochs=self.epochs,
                batch_size=2048,
                callbacks=[self.early_stop_system],
                validation_data=(x_test, y_test)
            )

        self.model.save(BASE_DIRECTORY / f'{self.name}.keras')