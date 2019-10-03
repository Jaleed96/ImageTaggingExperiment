import matplotlib.image as mpimg
import numpy as np
import os
from os import listdir
from os.path import isfile, join

def get_image(path):
    pic = mpimg.imread(path)
    pic = pic.astype('float32') / 255
    pic = np.expand_dims(pic, 0)

def get_bunch_img(folderpath):
    onlyfiles = [f for f in listdir(folderpath) if isfile(join(folderpath, f)) and os.path.splitext(f)[1] == ".jpg"]
    # batch_of_img = np.array((len(onlyfiles), -1, -1, 3))
    # batch_of_img = np.asarray([mpimg.imread(join(folderpath, pic)).astype('float32') / 255 for pic in onlyfiles])
    # batch_of_img = batch_of_img.reshape(4, -1, -1, 3)
    for i in range(len(onlyfiles)):
        batch_of_img = np.vstack((batch_of_img, mpimg.imread(join(folderpath, onlyfiles[i])).astype('float32') / 255))
    print(batch_of_img.shape)
    print(batch_of_img)