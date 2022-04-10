import numpy as np

def SGD(params, grads, architecture, learning_rate=1e-3, l2_param=0.03, decay_rate=0.95, period=0):
    """
    update parameters of each layer
    :param params: weights and biases of each layer
    :param grads: gradients of each layer
    :param architecture: architecture of network
    :param learning_rate: learning rate
    :param l2_param: l2 regularization parameter
    :param decay_rate: learning rate decay factor
    :param period: learning time
    :return: updated params
    """
    learning_rate = learning_rate * np.power(decay_rate, period)
    for index, layer in list(enumerate(architecture)):
        layer_index = index + 1
        params["W" + str(layer_index)] -= learning_rate * (grads["dW" + str(layer_index)]
                                                           + l2_param * params["W" + str(layer_index)])
        params["b" + str(layer_index)] -= learning_rate * grads["db" + str(layer_index)]
    return params