import numpy as np
import json


def init_params(architecture, seed=999):
    """
    initialize weights and biases according to the architecture
    :param architecture: architecture of network
    :param seed: random seed
    :return params: a dictionary including all the weights and biases
    """
    np.random.seed(seed)
    params = {}

    for index, layer in list(enumerate(architecture)):
        layer_index = index + 1
        layer_input_size = layer['input_dim']
        layer_output_size = layer['output_dim']

        params['W' + str(layer_index)] = np.random.rand(
            layer_input_size, layer_output_size)
        params['b' + str(layer_index)] = np.random.rand(
            1, layer_output_size)

    return params


"""
overwrite the default function in JSONEncoder
"""
class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_params(params, params_dir):
    """
    save params of network into the given directory
    :param params: a dictionary of numpy arrays of all params
    :param params_dir: directory of params to save
    """
    with open(params_dir, 'w') as f:
        json.dump(params, f, cls=NumpyArrayEncoder)

def read_params(params_dir):
    """
    read params of network
    :param params_dir: directory of saved params
    :return: params
    """
    params = {}
    with open(params_dir, 'r') as f:
        params_ = json.load(f)
        for key, value in params_.items():
            params[key] = np.asarray(value)
    return params