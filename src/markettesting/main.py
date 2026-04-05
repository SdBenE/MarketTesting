"""
Base main testing
"""
import warnings
import yfinance as yf
from markettesting.stock_models import StockModel
from markettesting.analysis import DataLog
from markettesting.formatting import flatten_from_yf

warnings.filterwarnings('ignore')

my_model = StockModel(sequence_length=100,model_import=True, time_period='1y')
# my_model.train_model(use_download=True,batch_size=256)

data_log_AAPL = DataLog(model=my_model, ticker_name='AAPL',use_download=False)
data_log_AAPL.determine_prediction_set(days=1000)

# data_set = yf.download('AAPL', period='1y')
# data_set = data_set[-100:]
# print(data_set)
# data_set = flatten_from_yf(data_set)
# print(data_set)
# prediction = my_model.get_prediction(data_set, return_close=True)
# print(prediction)