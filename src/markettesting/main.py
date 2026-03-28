"""
Base main testing
"""
import warnings
from stock_model import StockModel

warnings.filterwarnings('ignore')

my_model = StockModel(sequence_length=100)
my_model.train_model(use_download=True)
