"""
Base main testing
"""
import warnings
from markettesting.stock_models import StockModel
from markettesting.analysis import DataLog

warnings.filterwarnings('ignore')

my_model = StockModel(sequence_length=100,model_import=False)
my_model.train_model(use_download=True,batch_size=64)

data_log_AAPL = DataLog(model=my_model, ticker_name='AAPL',use_download=False)
data_log_AAPL.determine_prediction_set(days=90)