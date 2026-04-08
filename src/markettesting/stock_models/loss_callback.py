from keras.callbacks import Callback

class LossCallback(Callback):
    def __init__(self, ticker, threshold="10.0"):
        super().__init__()
        self.ticker = ticker
        self.threshold = threshold
        self.stop_training = False

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        if val_loss is not None and val_loss > self.threshold:
            print(f"{self.ticker} is unstable in training! {val_loss} exceeds goal threshold")
            print(f"Skipping {self.ticker}...")
            self.model.stop_training = True