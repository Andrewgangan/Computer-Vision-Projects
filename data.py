import traceback
import imageio
import glob
import numpy
import matplotlib.pyplot
import numpy as np


def decompress_data(image_f, label_f, out_f, n):
    """
    decompress MNIST data(images and labels), convert to .csv files and save
    :param image_f: directory of image
    :param label_f: directory of label
    :param out_f: directory of output
    :param n: number of samples
    :return: none
    """
    try:
        image_io = open(image_f, 'rb')
        label_io = open(label_f, 'rb')
        out_io = open(out_f, 'w')

        image_io.read(16)
        label_io.read(8)
        images = []

        for i in range(n):
            image = [ord(label_io.read(1))]
            for j in range(28 * 28):
                image.append(ord(image_io.read(1)))
            images.append(image)

        for image in images:
            out_io.write(",".join(str(pix) for pix in image) + '\n')

        image_io.close()
        label_io.close()
        out_io.close()
    except IOError as e:
        print("an IO error occurred...")
        traceback.print_exc()
    return

def get_data(data_dir):
    """
    read MNIST data(.csv) and reshape to a matrix and
    :param data_dir: directory of data to read
    :return X: images(28 * 28) array
    :return y: label array
    """
    try:
        data = np.loadtxt(data_dir, delimiter=',')
        y = data[:, 0]
        X = data[:, 1:28*28+1]
        return X, y
    except IOError as e:
        print("an IO error occurred...")
        traceback.print_exc()

def standardize_cols(X, **args):
    """
    Make each column of X be zero mean, std 1.
    If (mu, sigma) are default values (0, 1), they are computed from X
    :param X: 2d numpy array of input images, shape (n, 28 * 28)
    :return S: standardized X, shape (n, 28 * 28)
    :return mu: mean of each column, shape(1, 28 * 28)
    :return sigma: std of each column, shape(1, 28 * 28)
    """
    if 'mu' in args and 'sigma' in args:
        mu = args['mu']
        sigma = args['sigma']
    else:
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma[sigma == 0.0] = 1
    S = X - mu
    S = S / sigma
    return S, mu, sigma

def linearToBinary(y):
    """
    change y into one hot vectors
    :param y: numpy array of shape (n,), and y[i] ranges from 0 to c-1
    :return: yExpanded: numpy array of shape (n, c)
    """
    y = y.astype(int)
    n = y.shape[0]
    c = np.max(y) + 1
    yExpanded = np.zeros([n, c])
    for i in range(n):
        yExpanded[i][y[i]] = 1
    yExpanded = yExpanded * 2 - 1
    return yExpanded

def binaryToLinear(yExpanded):
    """
    change one hot vectors back into scalars of classes
    :param yExpanded: numpy array of shape (n, c)
    :return: y: numpy array of shape (n,), and y[i] ranges from 0 to c-1
    """
    return np.argmax(yExpanded, axis=1)


"""
decompress_data("data/train-images.idx3-ubyte",
                "data/train-labels.idx1-ubyte",
                "data/train_set.csv", 60000)
decompress_data("data/t10k-images.idx3-ubyte",
                "data/t10k-labels.idx1-ubyte",
                "data/test_set.csv", 10000)
"""