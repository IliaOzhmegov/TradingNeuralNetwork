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


tiny_narrow_window.plot()
# plt.title("Given 30 days as input, predict the 31st day.")
plt.show()


### MODEL SETTINGS
MAX_EPOCHS = 4

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


val_performance['Linear'] = multi_step_dense.evaluate(tiny_narrow_window.val)
performance['Linear'] = multi_step_dense.evaluate(tiny_narrow_window.test, verbose=0)

tiny_narrow_window.plot(multi_step_dense)
plt.show()
