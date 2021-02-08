import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from libs.config import SEQUENCES_PATH, SEQUENCES_PATH_


class Scaler:
    _PMIN = 0.044545454545454545
    _PMAX = 1 - _PMIN

    def __init__(self, min: float, max: float):
        self.__min = min
        self.__max = max

        self.__ymin = None
        self.__ymax = None
        self.__calculate_abs_marigins()

    def __calculate_abs_marigins(self):
        # LOCAL TESTING #
        # min = 4.5
        # max = 95.5
        #
        # pmin = 0.044545454545454545
        # pmax = 1 - pmin
        # LOCAL TESTING #

        min = self.__min
        max = self.__max

        pmin = Scaler._PMIN
        pmax = Scaler._PMAX

        A = np.array([[pmin, pmax], [pmax, pmin]])
        B = np.array([[min], [max]])
        invA = np.linalg.inv(A)
        tmp = invA @ B

        self.__ymin = tmp[1][0]
        self.__ymax = tmp[0][0]

    def get_marigins(self):
        return (self.__ymin, self.__ymax)

    def scale(self, y):
        ymin = self.__ymin
        ymax = self.__ymax

        return (y - ymin) / (ymax - ymin)


class WindowGenerator:
    def __init__(self,
                 train_df, val_df, test_df,
                 input_width=30, label_width=5, shift=5,
                 label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df[0].columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        #### LOCAL TESTING ####
        # total_window_size = 35
        # data = train_df.copy()
        #### LOCAL TESTING ####
        data = np.concatenate([np.array(window[:self.total_window_size]) for window in data])
        data = data.astype(np.float32)

        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=self.total_window_size,
            shuffle=True,
            batch_size=32,)

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

    def plot(self, model=None, plot_col='Close', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(3, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                try:
                    plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                                marker='X', edgecolors='k', label='Predictions',
                                c='#ff7f0e', s=64)
                except:
                    lab_n = len(self.label_indices)
                    pre_n = len(predictions[n, :, label_col_index])
                    min_n = min(lab_n, pre_n)
                    x = self.label_indices[(lab_n - min_n): lab_n]
                    y = predictions[n, :, label_col_index][(pre_n - min_n): pre_n]

                    plt.scatter(x, y,
                                marker='X', edgecolors='k', label='Predictions',
                                c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [days]')


# load dataset
def load_sequence_dataset():
    try:
        files = os.listdir(SEQUENCES_PATH)
        spath = SEQUENCES_PATH
    except FileNotFoundError:
        files = os.listdir(SEQUENCES_PATH_)
        spath = SEQUENCES_PATH_
    files.sort()

    ####
    i = 0
    ####

    sequences = []
    # day = 24*60*600
    # year = (365.2425)*day
    # file = files[0]
    for file in files:
        if not file.endswith(".csv"): continue

        ####
        i += 1
        if i % 100 == 0: print("Done:", i)
        ####

        try:
            df = pd.read_csv(spath + file)
            # date_time = pd.to_datetime(df.pop('Date'), format='%Y-%m-%d')
            # timestamp_s = date_time.map(datetime.datetime.timestamp)
            df = df[["Open", "High", "Low", "Close"]]
            # df['Year_sin'] = np.sin(timestamp_s * (2 * np.pi / year))
            # df['Year_cos'] = np.cos(timestamp_s * (2 * np.pi / year))
            sequences.append(df)
        except:
            print(file)
            continue

    return sequences
