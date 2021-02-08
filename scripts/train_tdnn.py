# Created by Ilia
import tensorflow as tf
import matplotlib.pyplot as plt

from libs.common import load_sequence_dataset
from libs.common import WindowGenerator


SAVEPATH = "plots/modelling/"

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


tiny_narrow_window = WindowGenerator(train_df=train_df, val_df=val_df, test_df=test_df,
                                     label_columns=['Close'], shift=1, label_width=1)
for example_inputs, example_labels in tiny_narrow_window.train.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')


tiny_narrow_window.plot()
plt.savefig(SAVEPATH + "tiny_narrow_window.png")
# plt.title("Given 30 days as input, predict the 31st day.")
# plt.show()


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

### TDNN MODEL
multi_step_dense = tf.keras.Sequential([
    # Shape: (time, features) => (time*features)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
    # Add back the time dimension.
    # Shape: (outputs) => (1, outputs)
    tf.keras.layers.Reshape([1, -1]),
])

print('Input shape:', tiny_narrow_window.example[0].shape)
print('Output shape:', multi_step_dense(tiny_narrow_window.example[0]).shape)

history = compile_and_fit(multi_step_dense, tiny_narrow_window)

plt.close()
plot_loss(history, title="TDNN - history")
plt.savefig(SAVEPATH + "TDNN_history.png")
# plt.show()

val_performance['TDNN'] = multi_step_dense.evaluate(tiny_narrow_window.val)
performance['TDNN'] = multi_step_dense.evaluate(tiny_narrow_window.test, verbose=0)

tiny_narrow_window.plot(multi_step_dense)
plt.savefig(SAVEPATH + "TDNN_prediction.png")
# plt.show()

### simpleRNN
rnn_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.SimpleRNN(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

print('Input shape:', tiny_narrow_window.example[0].shape)
print('Output shape:', rnn_model(tiny_narrow_window.example[0]).shape)

history = compile_and_fit(rnn_model, tiny_narrow_window)

plt.close()
plot_loss(history, "RNN history")
plt.savefig(SAVEPATH + "RNN_history.png")

val_performance['RNN'] = rnn_model.evaluate(tiny_narrow_window.val)
performance['RNN'] = rnn_model.evaluate(tiny_narrow_window.test, verbose=0)

tiny_narrow_window.plot(rnn_model)
plt.savefig(SAVEPATH + "prediction/RNN_prediction.png")

### CNN MODEL
conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=64,
                           kernel_size=(3,),
                           activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
])
print("Conv model on `conv_window`")
print('Input shape:', tiny_narrow_window.example[0].shape)
print('Output shape:', conv_model(tiny_narrow_window.example[0]).shape)

history = compile_and_fit(conv_model, tiny_narrow_window)
plt.close()
plot_loss(history, "CNN history")
plt.savefig(SAVEPATH + "CNN_history.png")
# plt.show()

val_performance['CNN'] = conv_model.evaluate(tiny_narrow_window.val)
performance['CNN'] = conv_model.evaluate(tiny_narrow_window.test, verbose=0)


tiny_narrow_window.plot(conv_model)
plt.savefig(SAVEPATH + "prediction/CNN_prediction.png")
# plt.show()

### LSTM MODEL

lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])


print('Input shape:', tiny_narrow_window.example[0].shape)
print('Output shape:', lstm_model(tiny_narrow_window.example[0]).shape)


history = compile_and_fit(lstm_model, tiny_narrow_window)
plt.close()
plot_loss(history, "LSTM history")
plt.savefig(SAVEPATH + "LSTM_history.png")

val_performance['LSTM'] = lstm_model.evaluate(tiny_narrow_window.val)
performance['LSTM'] = lstm_model.evaluate(tiny_narrow_window.test, verbose=0)
tiny_narrow_window.plot(lstm_model)
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
