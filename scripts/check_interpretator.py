import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpl_img
import tensorflow as tf
from tensorflow.keras.preprocessing import image

img_path = "plots/candles/967_0.png"  # no in the training/validation
img = image.load_img(img_path)
img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)
img_preprocessed = img_batch/255.

interpretator = tf.keras.models.load_model('models/interpretator')
prediction = interpretator.predict(img_preprocessed)
prediction