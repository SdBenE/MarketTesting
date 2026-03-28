import model_creation
import frontend
import warnings

warnings.filterwarnings('ignore')

frontend.intro()

# formatting.fileFormation(4, downLoad=True)
myModel = model_creation.StockModel(epochs=25, units=1000)
myModel.create_model(sequence_length=100)
myModel.train_model(use_download=True)