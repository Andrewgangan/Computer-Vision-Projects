from data import *
from model import full_forward, full_backward, calc_square_loss, calc_accuracy
from params import *
from SGD import SGD
import numpy as np
import matplotlib.pyplot as plt
import os

NN_ARCHITECTURE = [
    {"input_dim": 784, "output_dim": 200, "activator": "tanh"},
    {"input_dim": 200, "output_dim": 10, "activator": "tanh"}
]

MAX_ITER = 200              # maximum training iterations
BATCH_SIZE = 5000           # batch size
LEARNING_RATE = 1e-3        # learning rate
DECAY_RATE = 0.99           # learning rate decay factor
L2_PARAM = 0.0001            # l2-regularization parameter

PARAMS_DIR = 'params.npy'               # directory to save parameters of network
TRAIN_DATA_DIR = 'data/train_set.csv'   # directory of train data and labels
TEST_DATA_DIR = 'data/test_set.csv'     # directory of test data and labels


if not os.path.exists(TRAIN_DATA_DIR):
    decompress_data("data/train-images.idx3-ubyte",
                    "data/train-labels.idx1-ubyte",
                    TRAIN_DATA_DIR, 60000)
if not os.path.exists(TEST_DATA_DIR):
    decompress_data("data/t10k-images.idx3-ubyte",
                    "data/t10k-labels.idx1-ubyte",
                    "data/test_set.csv", 10000)

# if no initial parameters, then create it
if not os.path.exists(PARAMS_DIR):
    params = init_params(NN_ARCHITECTURE)
    save_params(params, PARAMS_DIR)
else:
    params = read_params(PARAMS_DIR)
print("params initialization finished...")


# read data, and then divide into train/validation/test sets
X1, y1 = get_data(TRAIN_DATA_DIR)
X_train, mu, sigma = standardize_cols(X1)
y_train = linearToBinary(y1)

X2, y2 = get_data(TEST_DATA_DIR)
X_test, _, _ = standardize_cols(X2, mu=mu, sigma=sigma)
y_test = linearToBinary(y2)
print("data loading and preprocessing finished...")

# recording loss and accuracy curves
train_loss = []
test_loss = []
test_accuracy = []

# training
for i in range(MAX_ITER):
    # randomly choose a batch of sample
    rand_index = np.floor(np.random.rand(BATCH_SIZE) * X_train.shape[0]).astype(int)

    X_batch = X_train[rand_index, :]
    y_batch = y_train[rand_index, :]

    # forward
    yhat_batch, cache = full_forward(X_batch, params, NN_ARCHITECTURE)
    loss_batch, _ = calc_square_loss(yhat_batch, y_batch)
    accuracy_batch = calc_accuracy(yhat_batch, y_batch)
    print("batch " + str(i + 1) + ": training loss = " + str(loss_batch) + ", training accuracy = " + str(accuracy_batch))
    train_loss.append(loss_batch)

    # calculate test loss
    yhat_test, _ = full_forward(X_test, params, NN_ARCHITECTURE)
    loss_test, _ = calc_square_loss(yhat_test, y_test)
    accuracy_test = calc_accuracy(yhat_test, y_test)
    test_loss.append(loss_test)
    test_accuracy.append(accuracy_test)

    # backward
    grads_batch = full_backward(yhat_batch, y_batch, cache, params, NN_ARCHITECTURE)

    # use SGD to update params
    params = SGD(params, grads_batch, NN_ARCHITECTURE, LEARNING_RATE, L2_PARAM, DECAY_RATE, period=i)

# save parameters
save_params(params, PARAMS_DIR)