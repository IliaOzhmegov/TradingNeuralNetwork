# Created by Ilia
import os
import numpy as np
import pandas as pd
import matplotlib.image as mpl_img

CANDLE_PATH = "plots/candles/"
TARGET_PATH = "data/sequences/"


def load_dataset():
    # load dataset
    files = os.listdir(CANDLE_PATH)
    files.sort()

    images = []
    target = []

    old_row = -1

    #####
    i = 0
    #####

    file = files[0]
    for file in files:
        if not file.endswith('png'):
            continue

        ####
        i += 1
        if i % 100 == 0: print("Done:", i)
        ####

        try:
            image = mpl_img.imread(CANDLE_PATH + file)
            image_mtx = np.asarray(image, dtype="float64")

            ifile, irow = file.split('.')[0].split('_')
            if old_row != int(irow):
                filename = TARGET_PATH + ifile + ".csv"
                df = pd.read_csv(filename)

            targets = np.asarray(df.iloc[int(irow)][["Open", "High", "Low", "Close"]])

            images.append(image_mtx)
            target.append(targets)

            old_row = irow
        except:
            print(file)
            continue

    images = np.array(images)
    target = np.array(target)
    return images, target

dataset = load_dataset()

np.save("data/translator_dataset/images.npy", dataset[0])
np.save("data/translator_dataset/labels.npy", dataset[1])


