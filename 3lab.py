import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.util import random_noise
from scipy.signal import convolve2d
from scipy.ndimage import median_filter


def chessboard():
    chess = (np.indices((128, 128)) // 16).sum(axis=0) % 2
    return chess


def additive_noise_calc(img, snr):
    disp = np.var(img) / 255 ** 2
    noise_img = random_noise(img/255, var=disp/snr)*255
    noise = noise_img - img + 128
    return noise_img, noise


if __name__ == "__main__":
    board = np.array([160, 96])
    orig_img = (board[chessboard()])
    plt.subplot(221)
    plt.imshow(orig_img, cmap='gray', vmin=0, vmax=255)
    disp_orig = np.var(orig_img)
    additive_noise_img, additive_noise = additive_noise_calc(orig_img, 1)
    plt.subplot(222)
    plt.imshow(additive_noise, cmap='gray', vmin=0, vmax=255)
    plt.subplot(223)
    plt.imshow(additive_noise_img, cmap='gray', vmin=0, vmax=255)
    plt.show()

    plt.subplot(221)
    plt.imshow(orig_img, cmap='gray', vmin=0, vmax=255)
    disp_add_noise = np.var(orig_img)/255**2
    additive_noise_img = random_noise(orig_img/255, var=disp_add_noise/10)*255
    additive_noise = additive_noise_img - orig_img + 128
    plt.subplot(222)
    plt.imshow(additive_noise, cmap='gray', vmin=0, vmax=255)
    plt.subplot(223)
    plt.imshow(additive_noise_img, cmap='gray', vmin=0, vmax=255)
    plt.show()
