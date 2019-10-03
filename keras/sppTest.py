import os
import tensorflow as tf
from keras import layers
from keras.models import Model
from spp import SpatialPyramidPooling
from keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)

print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

input_layer = layers.Input(name='input_layer', shape=(None, None, 3), dtype='float32')
conv1 = layers.Conv2D(32, (3, 3), border_mode='same', activation=tf.nn.relu)(input_layer)
maxp1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
drop1 = layers.Dropout(0.25)(maxp1)
conv2 = layers.Conv2D(64, (3, 3), border_mode='same', activation=tf.nn.relu)(drop1)
maxp2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
drop2 = layers.Dropout(0.25)(maxp2)
conv3 = layers.Conv2D(64, (3, 3), activation=tf.nn.relu)(drop2)
spp1 = SpatialPyramidPooling([1, 2, 4])(drop3)
dense1 = layers.Dense(512, activation=tf.nn.relu)(spp1)
drop3 = layers.Dropout(0.25)(dense1)
y_pred = layers.Dense(10, activation=tf.nn.softmax)(drop3)
Model(inputs=input_layer, outputs=y_pred).summary()
