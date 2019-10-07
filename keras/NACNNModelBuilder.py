import os
import tensorflow as tf
import keras
import numpy as np
from keras import layers
from keras.models import Model
from keras.models import Sequential
from spp import SpatialPyramidPooling
from keras.datasets import cifar100
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)

print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, 100)
y_test = keras.utils.to_categorical(y_test, 100)

model = Sequential()


model.add(layers.Conv2D(32, (3, 3), padding='same', input_shape=(None, None, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.GaussianNoise(0.1))
model.add(layers.Conv2D(64, (3, 3), padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.GaussianNoise(0.1))
model.add(layers.Conv2D(128, (3, 3)))
model.add(layers.Activation('relu'))
model.add(SpatialPyramidPooling([1, 2, 4]))
model.add(layers.GaussianNoise(0.1))
model.add(layers.Dense(512))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(100))
model.add(layers.Activation('softmax'))
model.summary()

# opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
learning_rate = 0.001
epochs = 100
decay_rate = learning_rate / epochs
momentum = 0.9
opt = keras.optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

dir_path = os.path.dirname(os.path.realpath(__file__))
# model_path = os.path.join(dir_path, 'sppRMSprop.h5')
model_path = os.path.join(dir_path, 'NACNN4.h5')
if(os.path.exists(model_path)):
    model.load_weights(model_path)

model.fit(x_train, y_train, batch_size=64, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)

model.save(model_path)


