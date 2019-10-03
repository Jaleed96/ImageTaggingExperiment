import numpy as np
from keras.models import load_model
from keras.datasets import cifar10
from keras.utils import np_utils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#model = load_model('my_model.h5')
#img = mpimg.imread('x.png')

# print(img.shape)
a = 32
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# data pre-processing

for _ in range(1, 10):
    b = np.random.randint(4, 50000)
    plt.imshow(X_train[b])
    plt.show()
print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
# # loss, accuracy = model.evaluate(X_test, y_test)

# print('\ntest loss: ', loss)
# print('\ntest accuracy: ', accuracy)
