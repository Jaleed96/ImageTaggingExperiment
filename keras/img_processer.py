import matplotlib.image as mpimg
import numpy as np
import os
from os import listdir
from os.path import isfile, join

def get_image(path):
    pic = mpimg.imread(path)
    pic = pic.astype('float32') / 255
    pic = np.expand_dims(pic, 0)
    return pic

def get_bunch_img(folderpath):
    return [join(folderpath, f) for f in listdir(folderpath) if isfile(join(folderpath, f)) and os.path.splitext(f)[1] == ".jpg"]