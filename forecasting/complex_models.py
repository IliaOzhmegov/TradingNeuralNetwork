# Created by Ilia
import tensorflow as tf
import matplotlib.pyplot as plt

from libs.common import load_sequence_dataset
from libs.common import WindowGenerator
from libs.common import compile_and_fit
from libs.common import plot_loss


SAVEPATH = "plots/modelling/complex_models/"

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

val_performance = {}
performance = {}

OUT_STEPS = 5
num_features = 4
### MULTI LINEAR MODEL
multi_linear_model = tf.keras.Sequential([
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_linear_model, standard_window)

plt.close()
plot_loss(history, title="foobar - history")
plt.savefig(SAVEPATH + "foobar_history.png")
# plt.show()

val_performance['foobar_models'] = multi_linear_model.evaluate(standard_window.val)
performance['foobar_models'] = multi_linear_model.evaluate(standard_window.test, verbose=0)

plt.close()
standard_window.plot(multi_linear_model)
plt.savefig(SAVEPATH + "TDNN_prediction.png")
