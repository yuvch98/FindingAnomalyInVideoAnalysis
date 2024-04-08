import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback

class PlotLearning(Callback):
  def on_train_begin(self, logs={}):
    self.acc = []
    self.val_acc = []
    self.loss = []
    self.val_loss = []

  def on_epoch_end(self, epoch, logs={}):
    self.acc.append(logs.get('accuracy'))
    self.val_acc.append(logs.get('val_accuracy'))
    self.loss.append(logs.get('loss'))
    self.val_loss.append(logs.get('val_loss'))

    plt.figure(figsize=(12, 6))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(self.acc, label='Train Accuracy')
    plt.plot(self.val_acc, label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(self.loss, label='Train Loss')
    plt.plot(self.val_loss, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()
