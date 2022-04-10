from data import *
from params import *
from train import PARAMS_DIR, TRAIN_DATA_DIR, TEST_DATA_DIR, NN_ARCHITECTURE
from model import full_forward, calc_accuracy, calc_square_loss

params = read_params(PARAMS_DIR)

if __name__ == '__main__':
    X1, _ = get_data(TRAIN_DATA_DIR)
    _, mu, sigma = standardize_cols(X1)

    X2, y2 = get_data(TEST_DATA_DIR)
    X_test, _, _ = standardize_cols(X2, mu=mu, sigma=sigma)
    y_test = linearToBinary(y2)

    yhat_test, _ = full_forward(X_test, params, NN_ARCHITECTURE)
    loss_test, _ = calc_square_loss(yhat_test, y_test)
    accuracy_test = calc_accuracy(yhat_test, y_test)
