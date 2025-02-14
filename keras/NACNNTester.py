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
import cifar100label as cfl
import img_processer as ip
import math

dir_path = os.path.dirname(os.path.realpath(__file__))

z_test_paths = ip.get_bunch_img(os.path.join(dir_path, 'images'))

(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

model_path = os.path.join(dir_path, 'NACNN4.h5')

x_test = x_test.astype('float32')
x_test /= 255

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
model.load_weights(model_path)

model.summary()

varfrom = 40
varto = 49
y_preds = model.predict(x_test[varfrom:varto])
# y_preds = np.argmax(y_preds, axis=-1)
lent = math.ceil(math.sqrt(varto - varfrom))
fig, axes = plt.subplots(lent, lent)
for i in range(varto - varfrom):
    row = (int)(i % lent)
    col = (int)(i / lent)
    sub = axes[row][col]
    sub.imshow(x_test[varfrom + i])
    reall = cfl.coarse_label[y_test[varfrom + i,0]]
    predl = cfl.translate_label([y_preds[i]])
    sub.set_title('Real: ' + reall + ' : Pred: ' + predl)
plt.show()

def batch_predit(paths, model):
    return np.array(list(map(lambda pic: model.predict(ip.get_image(pic)), paths))).reshape((-1,100))

def batch_display_label(paths, model):
    lent = len(paths)
    lent = math.ceil(math.sqrt(lent))
    fig, axes = plt.subplots(lent, lent)
    for i in range(len(paths)):
        img = mpimg.imread(paths[i])
        pic = ip.get_image(paths[i])
        row = (int)(i % lent)
        col = (int)(i / lent)
        sub = axes[row][col]
        sub.imshow(img)
        predl = cfl.translate_label(model.predict(pic))
        sub.set_title('Predition: ' + predl)
    plt.show()

batch_display_label(z_test_paths, model)