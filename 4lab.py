from matplotlib import pyplot as plt
import numpy as np
import random
from scipy.signal import convolve2d
from skimage.util import random_noise

maskT = np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]])
maskT_inv = np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1]])
noise = random_noise(np.zeros((128, 128)), var=0.05, clip=False)


def add_objects(img, mask, count=10):
    img = img.copy()
    h, w = img.shape
    H, W = mask.shape

    for i in range(count):
        ty, tx = random.randrange(h - H), random.randrange(w - W)
        img[ty:ty+H, tx:tx+W] += mask

    print(f"Добавлено {count} объектов")
    return img


def correlate(img, mask):
    mask = 1/np.linalg.norm(mask) * mask
    # print((mask ** 2).sum())
    return convolve2d(img, np.flip(mask))


def threshold(img, t):
    return np.ceil((img - t) / 255) * 255


if __name__ == '__main__':

    #####1#####
    corr1 = correlate(noise, maskT)
    res1 = threshold(corr1, 2.0)
    corr2 = correlate(noise, maskT_inv)
    res2 = threshold(corr2, 2.0)
    plt.subplot(231)
    plt.title('Фон (шум)')
    plt.imshow(noise, cmap='gray')
    plt.subplot(232)
    plt.title('Корреляция с mask1')
    plt.imshow(corr1, cmap='gray')
    plt.subplot(233)
    plt.title('Корреляция с mask2')
    plt.imshow(corr2, cmap='gray')
    plt.subplot(234)
    plt.title('Фон с объектами')
    plt.imshow(noise, cmap='gray')
    plt.subplot(235)
    plt.title('Пороговая с mask1')
    plt.imshow(res1, cmap='gray')
    plt.subplot(236)
    plt.title('Пороговая с mask2')
    plt.imshow(res2, cmap='gray')
    plt.show()
    ########2######
    img = add_objects(noise, maskT)
    corr1 = correlate(img, maskT)
    res1 = threshold(corr1, 2.0)
    corr2 = correlate(img, maskT_inv)
    res2 = threshold(corr2, 2.0)
    plt.subplot(231)
    plt.title('Фон (шум)')
    plt.imshow(noise, cmap='gray')
    plt.subplot(232)
    plt.title('Корреляция с mask1')
    plt.imshow(corr1, cmap='gray')
    plt.subplot(233)
    plt.title('Корреляция с mask2')
    plt.imshow(corr2, cmap='gray')
    plt.subplot(234)
    plt.title('Фон с объектами')
    plt.imshow(img, cmap='gray')
    plt.subplot(235)
    plt.title('Пороговая с mask1')
    plt.imshow(res1, cmap='gray')
    plt.subplot(236)
    plt.title('Пороговая с mask2')
    plt.imshow(res2, cmap='gray')
    plt.show()
    #######3######
    img2 = add_objects(noise, maskT)
    img2 = add_objects(img2, maskT_inv)
    corr1 = correlate(img2, maskT)
    res1 = threshold(corr1, 2.0)
    corr2 = correlate(img2, maskT_inv)
    res2 = threshold(corr2, 2.0)
    plt.subplot(231)
    plt.title('Фон (шум)')
    plt.imshow(noise, cmap='gray')
    plt.subplot(232)
    plt.title('Корреляция с mask1')
    plt.imshow(corr1, cmap='gray')
    plt.subplot(233)
    plt.title('Корреляция с mask2')
    plt.imshow(corr2, cmap='gray')
    plt.subplot(234)
    plt.title('Фон с объектами')
    plt.imshow(img2, cmap='gray')
    plt.subplot(235)
    plt.title('Пороговая с mask1')
    plt.imshow(res1, cmap='gray')
    plt.subplot(236)
    plt.title('Пороговая с mask2')
    plt.imshow(res2, cmap='gray')
    plt.show()
