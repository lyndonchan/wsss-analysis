from keras.callbacks import *

class EpochCallback(Callback):
    def __init__(self, completed_epochs=0):
        self.epoch_num = completed_epochs

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_num += 1
        logs['epoch_num'] = self.epoch_num
        logs['lr'] = K.get_value(self.model.optimizer.lr)
