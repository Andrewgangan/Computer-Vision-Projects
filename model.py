import numpy as np
import math

def forward(A_prev, W_curr, b_curr, activator='ReLU'):
    """
    forward propagate by only one layer
    :param A_prev: previous layer's output
    :param W_curr: current layer's weights
    :param b_curr: current layer's biases
    :param activator: current layer's activator
    :return A_curr: current layer's output
    :return Z_curr: current layer's output (unactivated)
    """
    Z_curr = affine(A_prev, W_curr, b_curr)

    if activator == 'ReLU':
        activation_func = ReLU
    elif activator == 'softmax':
        activation_func = softmax
    elif activator == 'tanh':
        activation_func = tanh
    else:
        raise Exception('Non-supported activation function')

    return activation_func(Z_curr), Z_curr


def full_forward(X, params, architecture):
    """
    predict y according to given data and network architecture
    :param X: input data
    :param params: weights and biases of each layer
    :param architecture: architecture of each layer
    :return A_curr: output
    :return cache: data of each layer, which is needed to calculate gradient
    """
    cache = {}          # store the value of each hidden layers' output
    A_curr = X

    nn_size = len(architecture)
    for index in range(nn_size):
        layer_index = index + 1
        layer = architecture[index]
        A_prev = A_curr

        activator_curr = layer['activator']
        W_curr = params['W' + str(layer_index)]
        b_curr = params['b' + str(layer_index)]
        A_curr, Z_curr = forward(A_prev, W_curr, b_curr, activator_curr)

        cache['A' + str(index)] = A_prev
        cache['Z' + str(layer_index)] = Z_curr

    return A_curr, cache

def calc_square_loss(yhat, y):
    """
    calculate square loss
    :param yhat: numpy array of shape (n, c)
                each row is the predicted prob of each class
    :param y:  numpy array of shape (n, c)
                each row is the actual one-hot vector of each class
    :return: square loss and the gradient to yhat
    """
    n, _ = y.shape
    loss = np.power(np.linalg.norm(yhat - y), 2) / n
    grad = 2 * (yhat - y) / n
    return loss, grad

def calc_cross_entropy_loss(yhat, y):
    """
    calculate cross-entropy loss
    :param yhat: numpy array of shape (n, c)
                each row is the predicted prob of each class
    :param y:  numpy array of shape (n, c)
                each row is the actual one-hot vector of each class
    :return: cross-entropy loss and the gradient to yhat
    """
    n, c = y.shape
    loss = 0
    log_func = np.vectorize(lambda x: 0.0 if x == 0 else math.log(x))
    delta = 1e-7        # to avoid nan
    log_yhat = log_func(yhat + delta)
    loss -= np.sum(y * log_yhat) / n
    """
    # l2 regularization
    for index, layer in enumerate(architecture):
        layer_index = index + 1
        W_curr = params['W' + str(layer_index)]
        loss += l2_lambda * np.linalg.norm(W_curr)
    """
    grad = np.divide(y, yhat + delta)
    return loss, grad

def calc_accuracy(yhat, y):
    """
    calculate the accuracy of the current prediction
    :param yhat: numpy array of shape (n, c)
                each row is the predicted prob of each class
    :param y:  numpy array of shape (n, c)
                each row is the actual one-hot vector of each class
    :return: accuracy
    """
    n = yhat.shape[0]
    yhat_ = np.argmax(yhat, axis=1)
    y_ = np.argmax(y, axis=1)
    return np.sum(yhat_ == y_) / n

def backward(dA_curr, W_curr, b_curr, Z_curr, A_prev, activator='ReLU'):
    """
    backward propagate by one layer
    :param dA_curr: loss's gradient to A_curr, numpy array of shape (n, m1)
    :param W_curr: current layer's weights, numpy array of shape (m, m1)
    :param b_curr: current layer's biases, numpy array of shape (m1,)
    :param Z_curr: current layer's output (unactivated), numpy array of shape (n, m1)
    :param A_prev: last layer's output, numpy array of shape (n, m)
    :param activator: name of current layer's activation function
    :return dA_prev: loss's gradient to A_prev
    :return dW_curr: loss's gradient to W_curr
    :return db_curr: loss's gradient to b_curr
    """
    if activator == 'ReLU':
        backward_activation_func = ReLU_backward
    elif activator == 'softmax':
        backward_activation_func = softmax_backward
    elif activator == 'tanh':
        backward_activation_func = tanh_backward
    else:
        raise Exception('Non-supported activation function')

    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    dW_curr = np.dot(A_prev.T, dZ_curr)
    db_curr = np.sum(dZ_curr, axis=0, keepdims=True)
    dA_prev = np.dot(dZ_curr, W_curr.T)

    return dA_prev, dW_curr, db_curr

def full_backward(yhat, y, cache, params, architecture):
    """
    calculate gradients of all layers using backward propagation
    :param yhat: numpy array of shape (n, c)
                each row is the predicted prob of each class
    :param y:  numpy array of shape (n, c)
                each row is the actual one-hot vector of each class
    :param cache: data of each layer, which is needed to calculate gradient
    :param params: weights and biases of each layer
    :param architecture: architecture of each layer
    :return: grads of each layer
    """
    grads = {}
    # m = y.shape[1]

    _, dA_prev = calc_square_loss(yhat, y)

    nn_size = len(architecture)
    for layer_idx_prev in range(nn_size - 1, -1, -1):
        layer_idx_curr = layer_idx_prev + 1
        layer = architecture[layer_idx_prev]
        activator_curr = layer['activator']

        dA_curr = dA_prev

        A_prev = cache['A' + str(layer_idx_prev)]
        Z_curr = cache['Z' + str(layer_idx_curr)]
        W_curr = params["W" + str(layer_idx_curr)]
        b_curr = params["b" + str(layer_idx_curr)]

        dA_prev, dW_curr, db_curr = backward(dA_curr, W_curr, b_curr, Z_curr, A_prev, activator_curr)

        grads['dW' + str(layer_idx_curr)] = dW_curr
        grads['db' + str(layer_idx_curr)] = db_curr

    return grads

def affine(X, W, b):
    """
    apply an affine transformation to data
    :param X: a numpy array of shape (N, d)
    :param W: weight, a numpy array of shape (d, m)
    :param b: bias, a numpy array of shape (m,)
    :return: a numpy array of shape (N, m)
    """
    return np.dot(X, W) + b

def ReLU(X):
    return np.maximum(0, X)

def ReLU_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def tanh(X):
    return np.tanh(X)

def tanh_backward(dA, Z):
    dZ = np.dot(dA, np.diag(np.sum(1 - np.tanh(Z) ** 2, axis=0)))
    return dZ

def softmax(X):
    X -= np.max(X, axis=1, keepdims=True)
    X = np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)
    return X

def softmax_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ = dZ * Z * (1 - Z)
    return dZ
