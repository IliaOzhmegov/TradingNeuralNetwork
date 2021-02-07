import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
tf.keras.backend.set_floatx('float64')
import numpy as np

iris, target = load_iris(return_X_y=True)

K.clear_session()
X = iris[:, :3]
y = iris[:, 3]
z = target
ds = tf.data.Dataset.from_tensor_slices((X, y, z)).shuffle(buffer_size=150).batch(32)

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d0 = Dense(16, activation='relu')
        self.d1 = Dense(32, activation='relu')
        self.d2_1 = Dense(1)
        self.d2_2 = Dense(4, activation='softmax')

    def call(self, x):
        x = self.d0(x)
        x = self.d1(x)
        y_1 = self.d2_1(x)
        y_2 = self.d2_2(x)
        return y_1, y_2

model = MyModel()

loss_objects = [tf.keras.losses.MeanAbsoluteError(), tf.keras.losses.SparseCategoricalCrossentropy()]
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

acc = tf.keras.metrics.Accuracy(name='categorical loss')
loss = tf.keras.metrics.MeanAbsoluteError()
#error = tf.keras.metrics.MeanAbsoluteError()

@tf.function
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        outputs = model(inputs)
        losses = [l(t, o) for l,o,t in zip(loss_objects, outputs, targets)]

    gradients = tape.gradient(losses, model.trainable_variables)
    #print(gradients)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    #optimizer.apply_gradients(zip(gradients[1], model.trainable_variables))
    return outputs


for epoch in range(50):
    for xx, yy, zz in ds: # what to do with zz, the categorical target?

        outs = train_step(xx, [yy,zz])

        res1 = acc.update_state(zz, np.argmax(outs[1], axis=1))
        res2 = loss.update_state(yy, outs[0])

    template = 'Epoch {:>2}, Accuracy: {:>5.2f}, MAE: {:>5.2f}'
    print(template.format(epoch+1, acc.result(), loss.result()))

    acc.reset_states()
    loss.reset_states()
