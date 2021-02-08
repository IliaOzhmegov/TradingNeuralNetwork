# Created by Ilia
import tensorflow as tf
import matplotlib.pyplot as plt

from libs.common import load_sequence_dataset
from libs.common import WindowGenerator


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


SAVEPATH = "plots/modelling/"
tiny_narrow_window.plot()
plt.savefig(SAVEPATH + "tiny_narrow_window.png")
# plt.title("Given 30 days as input, predict the 31st day.")
# plt.show()


### MODEL SETTINGS
MAX_EPOCHS = 20

def compile_and_fit(model, window, patience=2):
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


val_performance = {}
performance = {}


history = compile_and_fit(multi_step_dense, tiny_narrow_window)

def plot_loss(history):
    plt.close()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error [MSE]')
    plt.legend()
    plt.grid(True)


plot_loss(history)
plt.savefig(SAVEPATH + "TDNN_history.png")
# plt.show()

val_performance['TDNN'] = multi_step_dense.evaluate(tiny_narrow_window.val)
performance['TDNN'] = multi_step_dense.evaluate(tiny_narrow_window.test, verbose=0)

tiny_narrow_window.plot(multi_step_dense)
plt.savefig(SAVEPATH + "TDNN_prediction.png")
# plt.show()

### CNN MODEL
conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32,
                           kernel_size=(3,),
                           activation='relu'),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=1),
])
print("Conv model on `conv_window`")
print('Input shape:', tiny_narrow_window.example[0].shape)
print('Output shape:', conv_model(tiny_narrow_window.example[0]).shape)

history = compile_and_fit(conv_model, tiny_narrow_window)
plt.close()
plot_loss(history)
plt.savefig(SAVEPATH + "CNN_history.png")
# plt.show()

val_performance['CNN_prediction'] = conv_model.evaluate(tiny_narrow_window.val)
performance['CNN_prediction'] = conv_model.evaluate(tiny_narrow_window.test, verbose=0)

tiny_narrow_window.plot(conv_model)
plt.savefig(SAVEPATH + "CNN_prediction.png")
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

val_performance['LSTM'] = lstm_model.evaluate(tiny_narrow_window.val)
performance['LSTM'] = lstm_model.evaluate(tiny_narrow_window.test, verbose=0)
