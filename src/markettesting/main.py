"""
Base main testing
"""
import warnings
import formatting
from StockModel import StockModel

warnings.filterwarnings('ignore')

# formatting.file_formation(4, download=True)
my_model = StockModel(epochs=25, units=1000, sequence_length=100)
my_model.train_model(use_download=True)
