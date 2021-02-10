# Created by Ilia
import numpy as np
import tensorflow as tf
# from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
# from tensorflow.keras import Model
from tensorflow.keras import models
# from tensorflow.keras.utils import to_categorical
# import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
tf.keras.backend.set_floatx('float64')

IMAGES_PATH = "../data/translator_dataset/images.npy"
TARGETS_PATH = "../data/translator_dataset/targets.npy"

images  = np.load(IMAGES_PATH).astype(np.float32)
targets = np.load(TARGETS_PATH, allow_pickle=True).astype(np.float32)

prop = 0.8
sep = int(images.shape[0] * prop)

train_dataset = images[:sep]
train_targets = targets[:sep]

test_dataset = images[sep:]
test_targets = targets[sep:]


# class Interpretator(Model):
#     def __init__(self):
#         super(Interpretator, self).__init__()
#         self.c0 = layers.Conv2D(32, (3, 3), activation="relu", input_shape=(200, 34, 3))
#         self.p0 = layers.MaxPool2D((2, 2))
#         self.c1 = layers.Conv2D(64, (3, 3), activation="relu")
#
#         # self.p1 = layers.MaxPool2D((2, 2))
#         # self.c2 = layers.Conv2D(64, (3, 3), activation="relu")
#
#         self.f2 = layers.Flatten()
#         self.fc = layers.Dense(64, activation="relu")
#
#         self.d3 = Dense(4)
#
#     def call(self, x):
#         x = self.c0(x)
#         x = self.p0(x)
#
#         x = self.c1(x)
#         x = self.p0(x)
#
#         x = self.f2(x)
#         x = self.fc(x)
#         x = self.d3(x)
#
#         return x

interpretator = models.Sequential()
interpretator.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(200, 34, 3)))
interpretator.add(layers.MaxPool2D((2, 2)))
interpretator.add(layers.Conv2D(64, (3, 3), activation="relu"))
interpretator.add(layers.MaxPool2D((2, 2)))
interpretator.add(layers.Flatten())
interpretator.add(layers.Dense(64, activation="relu"))
interpretator.add(layers.Dense(4))
print(interpretator.summary())

interpretator.compile(optimizer='adam',
                      loss=tf.keras.losses.MeanAbsoluteError(),
                      metrics=['mse'])
history = interpretator.fit(train_dataset, train_targets, epochs=10,
                            validation_data=(test_dataset, test_targets))


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title("Interpretator Training")
    plt.legend()
    plt.grid(True)


plt.figure(figsize=(12, 8))
plot_loss(history)
plt.savefig("plots/training_history_of_interpretator.png")
# plt.show()

interpretator.save("../models/interpretator")
