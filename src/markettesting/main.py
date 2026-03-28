import model_creation
import frontend
import warnings
import formatting

warnings.filterwarnings('ignore')

frontend.intro()

formatting.file_formation(4, download=True)
myModel = model_creation.StockModel(epochs=25, units=1000)
myModel.create_model(sequence_length=100)
myModel.train_model(use_download=True)