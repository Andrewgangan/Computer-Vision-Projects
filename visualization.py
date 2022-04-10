import numpy as np
from params import *
from train import PARAMS_DIR
import matplotlib.pyplot as plt

params = read_params(PARAMS_DIR)

W1 = params['W1'].reshape(28, 28, -1).transpose(2, 0, 1)
plt.rcParams['image.cmap'] = 'gray'

count = 0
for i in range(32):
    for j in range(8):
        plt.subplot(80, 10, count + 1)
        img = W1[count, :, :]
        img = 255.0 * (img - np.min(img)) / (np.max(img) - np.min(img))
        plt.imshow(img.astype('uint8'))
        plt.gca().axis('off')
        count += 1

plt.savefig('W1.png')

W2 = params['W2'].reshape(16, 16, -1).transpose(2, 0, 1)
plt.rcParams['image.cmap'] = 'gray'

count = 0
for i in range(2):
    for j in range(5):
        plt.subplot(2, 5, count + 1)
        img = W2[count, :, :]
        img = 255.0 * (img - np.min(img)) / (np.max(img) - np.min(img))
        plt.imshow(img.astype('uint8'))
        plt.gca().axis('off')
        count += 1

plt.savefig('W2.png')