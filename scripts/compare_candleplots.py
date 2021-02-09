import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.dates as mpdates
import mplfinance as mpf
from mplfinance.original_flavor import candlestick_ohlc
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing import timeseries_dataset_from_array


from libs.common import load_sequence_dataset
from libs.common import WindowGenerator
from libs.common import Scaler

df = pd.read_csv("data/SP500.csv", index_col=0, parse_dates=True)

n_predictors = 30
n_to_predict = 5

lower = int(len(df) * 0.9)  # from validation
upper = int(len(df) * 0.95)  # from validation
sep = lower + np.random.randint(200, upper - lower)

columns = ["Open", "High", "Low", "Close"]
example = np.array(df.iloc[sep:sep+n_predictors][columns])
scaler = Scaler(min=example.min(), max=example.max())
example = scaler.scale(example)

# predict candles
tdnn = tf.keras.models.load_model('models/TDNN')
tensored_example = tf.convert_to_tensor(np.array([example]), dtype=tf.float32)
predicted = np.array(tdnn.predict(tensored_example))
predicted = pd.DataFrame(predicted[0], columns=columns)

# draw predicted
predicted.insert(0, "Date", np.arange(0,5))

alpha = 0.5
width = 0.8
fig, ax = plt.subplots()
# plotting the data
candlestick_ohlc(ax, predicted.values, width=width,
                 colorup='green', colordown='red',
                 alpha=alpha)

plt.show()



# draw original
from_ = sep + n_predictors
to_ = from_ + n_to_predict

original = df.iloc[from_:to_]
original = original[columns]
original = scaler.scale(original)
original.insert(0, "Date", np.arange(0, 5))


fig, ax = plt.subplots()
# plotting the data
candlestick_ohlc(ax, original.values, width=width,
                 colorup='green', colordown='red',
                 alpha=alpha)

plt.show()


