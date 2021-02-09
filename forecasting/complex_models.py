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

