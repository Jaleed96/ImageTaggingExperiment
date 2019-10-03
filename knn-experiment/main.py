import knn as knn_algo
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pickle
from PIL import Image
from IPython.display import display

basedir_data = './data/'

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def default_label_fn(i, original):
    return original

def show_img(img_arr, label_arr, meta, index, label_fn=default_label_fn):
    """
        Given a numpy array of image from CIFAR-10 labels this method transform the data so that PIL can read and show
        the image.
        Check here how CIFAR encodes the image http://www.cs.toronto.edu/~kriz/cifar.html
    """
    
    one_img = img_arr[index, :]
    # Assume image size is 32 x 32. First 1024 px is r, next 1024 px is g, last 1024 px is b from the (r,g b) channel
    r = one_img[:1024].reshape(32, 32)
    g = one_img[1024:2048].reshape(32, 32)
    b = one_img[2048:]. reshape(32, 32)
    rgb = np.dstack([r, g, b])
    img = Image.fromarray(np.array(rgb), 'RGB')
    #img.show()
    print(label_fn(index, meta[label_arr[index][0]].decode('utf-8')))

def main():
    X = unpickle(basedir_data + 'train')
    img_data = X[b'data']
    img_label_orig = img_label = X[b'fine_labels']
    img_label = np.array(img_label).reshape(-1, 1)

    test_X = unpickle(basedir_data + 'test')
    test_data = test_X[b'data']
    test_label = test_X[b'fine_labels']
    test_label = np.array(test_label).reshape(-1, 1)
    
    batch = unpickle(basedir_data + 'meta')
    meta = batch[b'fine_label_names']

    sample_img_data = img_data[0:100, :]
    sample_test_data = test_data[0:100, :]

    neighbors = KNeighborsClassifier(n_neighbors=3, algorithm='brute').fit(img_data, img_label_orig)
    pred = neighbors.predict(sample_test_data)

    def pred_label_fn(i, original):
        return original + '::' + meta[pred[i]].decode('utf-8')

    for i in range(0, len(pred)):
        show_img(sample_test_data, test_label, meta, i, label_fn=pred_label_fn)

    


main()