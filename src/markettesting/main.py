"""
Base main testing
"""
import warnings
import yfinance as yf
from markettesting.stock_model import StockModel
from markettesting.formatting import flatten_from_yf

warnings.filterwarnings('ignore')

my_model = StockModel(sequence_length=100,model_import=True)
# my_model.train_model(use_download=True,batch_size=256)

data_set = yf.download('AAPL', period='1y')
data_set = data_set[-100:]
print(data_set)
data_set = flatten_from_yf(data_set)
print(data_set)
prediction = my_model.get_prediction(data_set)
print(prediction)