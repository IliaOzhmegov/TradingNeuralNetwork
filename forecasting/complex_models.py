# Created by Ilia
import tensorflow as tf
import matplotlib.pyplot as plt

from libs.common import load_sequence_dataset
from libs.common import WindowGenerator
from libs.common import compile_and_fit
from libs.common import plot_loss


if __name__ == "__main__":
    SAVEPATH = "../plots/modelling/complex_models/"
    MODELPATH = "../models/"
else:
    SAVEPATH = "plots/modelling/complex_models/"
    MODELPATH = "models/"

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


standard_window.plot()
plt.savefig(SAVEPATH + "standard_window.png")
# plt.title(""Provided with 30 days as input, preidct the next 5 days.")
# plt.show()

val_performance = {}
performance = {}

time = 5
features = 4



# MULTI LAYER, MULTI OUTPUT MODEL
multi_step_dense = tf.keras.Sequential([
    # tf.keras.layers.Lambda(lambda x: x[:, -1:, :]), tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(time * features, kernel_initializer=tf.initializers.glorot_normal()),
    tf.keras.layers.Reshape([time, features])
])

print('Input shape:', standard_window.example[0].shape)
print('Output shape:', multi_step_dense(standard_window.example[0]).shape)

history = compile_and_fit(multi_step_dense, standard_window)

plt.close()
plot_loss(history, title="TDNN - history")
plt.savefig(SAVEPATH + "TDNN_history.png")
# plt.show()

val_performance['TDNN'] = multi_step_dense.evaluate(standard_window.val)
performance['TDNN'] = multi_step_dense.evaluate(standard_window.test, verbose=0)

plt.close()
standard_window.plot(multi_step_dense)
plt.savefig(SAVEPATH + "prediction/TDNN_close_prediction.png")

multi_step_dense.save(MODELPATH + "TDNN")


### simpleRNN
rnn_model = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.SimpleRNN(32, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(time*features, kernel_initializer=tf.initializers.zeros()),
    tf.keras.layers.Reshape([time, features])
])


history = compile_and_fit(rnn_model, standard_window)

rnn_model.save(MODELPATH + "RNN")

plt.close()
plot_loss(history, "RNN history")
plt.savefig(SAVEPATH + "RNN_history.png")

val_performance['RNN'] = rnn_model.evaluate(standard_window.val)
performance['RNN'] = rnn_model.evaluate(standard_window.test, verbose=0)

standard_window.plot(rnn_model)
plt.savefig(SAVEPATH + "prediction/RNN_close_prediction.png")


### CNN MODEL

conv_w = 3
multi_conv_model = tf.keras.Sequential([
    tf.keras.layers.Lambda(lambda x: x[:, -conv_w:, :]),
    tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(conv_w)),
    tf.keras.layers.Dense(time*features, kernel_initializer=tf.initializers.zeros()),
    tf.keras.layers.Reshape([time, features])
])


print("Conv model on `standard_window`")
print('Input shape:', standard_window.example[0].shape)
print('Output shape:', multi_conv_model(standard_window.example[0]).shape)

history = compile_and_fit(multi_conv_model, standard_window)
plt.close()
plot_loss(history, "CNN history")
plt.savefig(SAVEPATH + "CNN_history.png")
# plt.show()

val_performance['CNN'] = multi_conv_model.evaluate(standard_window.val)
performance['CNN'] = multi_conv_model.evaluate(standard_window.test, verbose=0)

standard_window.plot(multi_conv_model)
plt.savefig(SAVEPATH + "prediction/CNN_close_prediction.png")
# plt.show()
multi_conv_model.save(MODELPATH + "CNN")


### LSTM MODEL

lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(time * features, kernel_initializer=tf.initializers.zeros()),
    tf.keras.layers.Reshape([time, features])
])


print('Input shape:', standard_window.example[0].shape)
print('Output shape:', lstm_model(standard_window.example[0]).shape)


history = compile_and_fit(lstm_model, standard_window)

plt.close()
plot_loss(history, "LSTM history")
plt.savefig(SAVEPATH + "LSTM_history.png")

val_performance['LSTM'] = lstm_model.evaluate(standard_window.val)
performance['LSTM'] = lstm_model.evaluate(standard_window.test, verbose=0)
standard_window.plot(lstm_model)
plt.savefig(SAVEPATH + "prediction/LSTM_close_prediction.png")
lstm_model.save(MODELPATH + "LSTM")

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

plt.ylabel('mean_absolute_error [All]')
plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.ylim((0.13, 0.14))
plt.xticks(ticks=x, labels=performance.keys(),
           rotation=45)
_ = plt.legend()
plt.savefig(SAVEPATH + "performance.png")


for name, value in performance.items():
    print(f'{name:8s}: {value[1]:0.4f}')
