# Created by Ilia
import tensorflow as tf
import matplotlib.pyplot as plt

from libs.common import load_sequence_dataset
from libs.common import WindowGenerator
from libs.common import compile_and_fit
from libs.common import plot_loss


# SAVEPATH = "plots/modelling/complex_models/"
SAVEPATH = "C:/Users/mryus/PycharmProjects/TradingNeuralNetwork/plots/modelling/complex_models/"
# SAVE_MODEL_PATCH = "C:/Users/mryus/PycharmProjects/TradingNeuralNetwork/models"

dataset = load_sequence_dataset()

n = len(dataset)
n_train, n_val = int(0.7*n), int(0.9*n)
train_df = dataset[:n_train]
val_df   = dataset[n_train:n_val]
test_df  = dataset[n_val:]



standard_window = WindowGenerator(train_df=train_df, val_df=val_df, test_df=test_df)

for example_inputs, example_labels in standard_window.train.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')


"""
standard_window.plot()
plt.savefig(SAVEPATH + "standard_window.png")
# plt.title(""Provided with 30 days as input, preidct the next 5 days.")
# plt.show()
"""

### MODEL SETTINGS
MAX_EPOCHS = 20

def compile_and_fit(model, window, patience=5):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history

def plot_loss(history, title):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Error [MSE]')
    plt.legend()
    plt.grid(True)


val_performance = {}
performance = {}

time = 5
features = 4

multi_linear_model = tf.keras.Sequential([
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(time*features,
                          kernel_initializer=tf.initializers.zeros),
    tf.keras.layers.Reshape([time, features])
])

history = compile_and_fit(multi_linear_model, standard_window)

#tf.keras.models.save_model("C:/Users/mryus/PycharmProjects/TradingNeuralNetwork/models/complex_multi_linear.pb")

plt.close()
plot_loss(history, title="foobar - history")
plt.savefig(SAVEPATH + "foobar_history.png")
# plt.show()

val_performance['foobar_models'] = multi_linear_model.evaluate(standard_window.val)
performance['foobar_models'] = multi_linear_model.evaluate(standard_window.test, verbose=0)

plt.close()
standard_window.plot(multi_linear_model)
plt.savefig(SAVEPATH + "TDNN_prediction.png")





### simpleRNN
rnn_model = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.SimpleRNN(32, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(time*features,
                          kernel_initializer=tf.initializers.zeros),
    tf.keras.layers.Reshape([time, features])
])


history = compile_and_fit(rnn_model, standard_window)

#tf.keras.models.save_model("C:/Users/mryus/PycharmProjects/TradingNeuralNetwork/models/complex_RNN.pb")

plt.close()
plot_loss(history, "RNN history")
plt.savefig(SAVEPATH + "RNN_history.png")

val_performance['RNN'] = rnn_model.evaluate(standard_window.val)
performance['RNN'] = rnn_model.evaluate(standard_window.test, verbose=0)

standard_window.plot(rnn_model)
plt.savefig(SAVEPATH + "prediction/RNN_prediction.png")


"""


### CNN MODEL
conv_model = tf.keras.Sequential([
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Conv1D(filters=64,
                           kernel_size=(3,),
                           activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(time * features,
                          kernel_initializer=tf.initializers.zeros),
    tf.keras.layers.Reshape([time, features])
])
print("Conv model on `conv_window`")
print('Input shape:', standard_window.example[0].shape)
print('Output shape:', conv_model(standard_window.example[0]).shape)

history = compile_and_fit(conv_model, standard_window)
plt.close()
plot_loss(history, "CNN history")
plt.savefig(SAVEPATH + "CNN_history.png")
# plt.show()

val_performance['CNN'] = conv_model.evaluate(standard_window.val)
performance['CNN'] = conv_model.evaluate(standard_window.test, verbose=0)


standard_window.plot(conv_model)
plt.savefig(SAVEPATH + "prediction/CNN_prediction.png")
# plt.show()

"""

### LSTM MODEL

lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(time * features,
                          kernel_initializer=tf.initializers.zeros),
    tf.keras.layers.Reshape([time, features])
])


print('Input shape:', standard_window.example[0].shape)
print('Output shape:', lstm_model(standard_window.example[0]).shape)


history = compile_and_fit(lstm_model, standard_window)

#tf.keras.models.save_model("C:/Users/mryus/PycharmProjects/TradingNeuralNetwork/models/complex_LSTM.pb")

plt.close()
plot_loss(history, "LSTM history")
plt.savefig(SAVEPATH + "LSTM_history.png")

val_performance['LSTM'] = lstm_model.evaluate(standard_window.val)
performance['LSTM'] = lstm_model.evaluate(standard_window.test, verbose=0)
standard_window.plot(lstm_model)
plt.savefig(SAVEPATH + "prediction/LSTM_prediction.png")

### Performance
import numpy as np
plt.close()
plt.close()
plt.close()
plt.close()
plt.figure(figsize=(12, 8))
x = np.arange(len(performance))
width = 0.3
metric_name = 'mean_absolute_error'
metric_index = lstm_model.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in val_performance.values()]
test_mae = [v[metric_index] for v in performance.values()]

plt.ylabel('mean_absolute_error [Close]')
plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=performance.keys(),
           rotation=45)
_ = plt.legend()
plt.savefig(SAVEPATH + "performance.png")