import model_creation
import frontend
import warnings

warnings.filterwarnings('ignore')

frontend.intro()

# formatting.fileFormation(periodYears=4, downLoad=False)

# entry = input(f'<Entry> : ')

# while entry != 'exit' or entry != 'q' or entry != 'quit':
#     frontend.checkInput(entry)
#     entry = input(f'<Entry> : ')

# while entry != 'exit' or entry != 'q' or entry != 'quit':
#     frontend.checkInput(entry)
#     entry = input(f'<Entry> : ')

# myDataLog = dataLog.DataLog()
# myDataLog.importDataLog()
# myDataLog.addElement('NVDA', '30d')

# dataFinder.createData('2021-01-01', '2025-01-01')
# myData = dataFinder.parseData('AAPL')

# formatting.graphData('ADP')
# formatting.graphData('AAPL')

# print("\n")


# formatting.fileFormation(4, downLoad=True)

myModel = model_creation.StockModel(epochs=25, units=1000)
myModel.create_model(sequence_length=100)
myModel.train_model(use_download=True)


# print(f"NP DATASET: {myData}\n")

# print(f"TF DATASET: {dataset}")
