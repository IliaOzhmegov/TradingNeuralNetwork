# Created by Ilia
import time
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.dates as mpdates
from mplfinance.original_flavor import candlestick_ohlc

DBPATH = "../data/SP500.csv"
PATH = "../plots/candle_plot.png"
CANDLE_PATH = "../plots/candles/"


def add_gauss_noise(img):
    row, col, ch = img.shape
    mean = 0
    var = 3
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = img + gauss
    return noisy.astype('uint8')


def draw_candle_plot(df, i, n_predictors):
    # window data frame
    wdf = df.iloc[i:i + n_predictors, :].copy()
    t0 = wdf.Date[i]
    wdf["Date"] = np.arange(t0, t0 + n_predictors)

    # to make model more robust
    alpha = np.random.random() * 0.5 + 0.5
    width = np.random.random() * 0.6 + 0.3

    fig, ax = plt.subplots()
    # plotting the data
    candlestick_ohlc(ax, wdf.values, width=width,
                     colorup='green', colordown='red',
                     alpha=alpha)
    # Removing litter around the plot
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    fig.tight_layout()

    fig.set_size_inches(12, 2)
    plt.savefig(PATH, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)


def slice_candle_plot(i, n_predictors):
    # Calculated manually
    y = 5
    h = 200
    x = 15
    w = 1137

    img = cv2.imread(PATH)
    crop_img = img[y:y+h, x:x+w]

    # I want to have only 20% of noisy candles to improve robustness
    if np.random.randint(10) >= 8:
        crop_img = add_gauss_noise(crop_img)

    # i-th window j-th candle
    pace = np.round(w / n_predictors).astype("int")
    delta = pace - 4
    for j in range(n_predictors):
        xshift = j * pace - (np.random.randint(3) - 1)  # improve robustness
        candle = crop_img[:, xshift:xshift+delta]

        pathname = CANDLE_PATH + str(i) + "_" + str(j) + ".png"
        try:
            cv2.imwrite(pathname, candle)
        except cv2.error:
            print(pathname)


if __name__ == "__main__":
    df = pd.read_csv(DBPATH)
    df = df[['Date', 'Open', 'High', 'Low', 'Close']]

    # convert into datetime object
    df['Date'] = pd.to_datetime(df['Date'])
    # apply map function
    df['Date'] = df['Date'].map(mpdates.date2num)  # I actually don't need real dates

    n_predictors = 30
    start = 0
    # n = df.shape[0]
    n = 1000 + n_predictors

    for i in range(start, n - n_predictors):
        # it implicitly saves plot on SSD
        draw_candle_plot(df=df, i=i, n_predictors=n_predictors)
        time.sleep(0.1)
        # it implicitly the plot and slices it
        slice_candle_plot(i=i, n_predictors=n_predictors)

        if i % 10 == 0: print("Done:", i)

