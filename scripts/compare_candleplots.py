import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from mplfinance.original_flavor import candlestick_ohlc

from libs.common import Scaler

df = pd.read_csv("data/SP500.csv", index_col=0, parse_dates=True)

n_predictors = 30
n_to_predict = 5

lower = int(len(df) * 0.7)  # from validation
upper = int(len(df) * 0.9)  # from validation
sep = lower + np.random.randint(200, upper - lower)

columns = ["Open", "High", "Low", "Close"]
trend_before = np.array(df.iloc[sep:sep + n_predictors][columns])
scaler = Scaler(min=trend_before.min(), max=trend_before.max())
trend_before = scaler.scale(trend_before)

# KNOWN TREND
trend_before_relative = pd.DataFrame(trend_before, columns=columns)
trend_before_relative.insert(0, "Date", np.arange(-30, 0))


# EXPECTED TREND
from_ = sep + n_predictors
to_ = from_ + n_to_predict

expected_trend_after = df.iloc[from_:to_]
expected_trend_after = expected_trend_after[columns]
expected_trend_after = scaler.scale(expected_trend_after)
expected_trend_after.insert(0, "Date", np.arange(0, 5))


# PREDICT


# PLOT PARAMETERS
alpha_before = 0.9
alpha_after = 0.5
width_before = 0.8
width_after = 0.5

# DRAWING COMBINED PLOT
def draw_prediction(model_path='models/TDNN'):
    model = tf.keras.models.load_model(model_path)
    tensored_example = tf.convert_to_tensor(np.array([trend_before]), dtype=tf.float32)
    predicted_trend_after = np.array(model.predict(tensored_example))
    predicted_trend_after = pd.DataFrame(predicted_trend_after[0], columns=columns)
    predicted_trend_after.insert(0, "Date", np.arange(0, 5))

    model_name = model_path.split('/')[-1]

    fig, ax = plt.subplots(nrows=2, ncols=1)
    fig.set_size_inches(10, 5)
    ax[0].title.set_text('Original')
    ax[1].title.set_text('Predicted by ' + model_name)
    candlestick_ohlc(ax[0], trend_before_relative.values, width=width_before,
                     colorup='green', colordown='red',
                     alpha=alpha_before)

    candlestick_ohlc(ax[0], expected_trend_after.values, width=width_after,
                     colorup='green', colordown='red',
                     alpha=alpha_after)

    candlestick_ohlc(ax[1], trend_before_relative.values, width=width_before,
                     colorup='green', colordown='red',
                     alpha=alpha_before)

    candlestick_ohlc(ax[1], predicted_trend_after.values, width=width_after,
                     colorup='green', colordown='red',
                     alpha=alpha_after)
    fig.tight_layout()
    plt.savefig("plots/predictions/" + model_name + "_prediction.png")
    plt.show()


draw_prediction()
draw_prediction(model_path="models/RNN")
conv_w = 3
draw_prediction(model_path="models/CNN")
draw_prediction(model_path="models/LSTM")
