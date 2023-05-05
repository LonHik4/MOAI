from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import skimage.io


def entropy(arr):
    _, counts = np.unique(arr, return_counts=True)
    counts = counts / counts.sum()
    return -(counts * np.log2(counts)).sum()


def predictor(i, j, y, number):
    if number == 1:
        if i == 0 and j == 0:
            return 0
        if j == 0:
            return y[i - 1][-1]
        return y[i][j - 1]
    if number == 2:
        if i == 0 or j == 0:
            return 0
        return int((y[i][j - 1] + y[i - 1][j]) / 2)


def MyDifCode(img, e, number):
    f = np.zeros(img.shape)
    y = np.zeros(img.shape)
    q = np.zeros(img.shape)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            p = predictor(i, j, y, number)
            f[i][j] = img[i][j] - p
            q[i][j] = np.sign(f[i][j]) * ((abs(f[i][j]) + e) // (2 * e + 1))
            y[i][j] = p + q[i][j] * (2 * e + 1)
            if np.max(img[i][j] - y[i][j]) <= e:
                continue
            else:
                print("Error")

    return q, f


def MyDifDeCode(q, e, number):
    y = np.zeros(q.shape)
    for i in range(q.shape[0]):
        for j in range(q.shape[1]):
            p = predictor(i, j, y, number)
            y[i][j] = p + q[i][j] * (2 * e + 1)

    return y


if __name__ == "__main__":

    img = cv.imread('04_boat.tif', cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    # x = range(0, 51, 5)
    # y1 = [entropy(MyDifCode(img, e, 1)[0]) for e in x]
    # y2 = [entropy(MyDifCode(img, e, 2)[0]) for e in x]
    # plt.plot(x, y1, color="green", label="предсказатель 1")
    # plt.plot(x, y2, color="red", label="предсказатель 2")
    # plt.legend()
    # plt.show()

    # e = 5, 10, 20, 40
    # fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    # axes = axes.flatten()
    #
    # for i in range(4):
    #     q, _ = MyDifCode(img, e[i], 1)
    #     y = MyDifDeCode(q, e[i], 1)
    #     axes[i].imshow(y, cmap="gray")
    #     axes[i].set_title(f"Восстановление e={e[i]}")
    # plt.show()

    e = 0, 0, 5, 10
    fig, axes = plt.subplots(2, 2)
    axes = axes.flatten()

    for i in range(4):
        q, f = MyDifCode(img, e[i], 1)

        if i == 0:
            s = f
            desc = "неквантованный"
        else:
            s = q
            desc = "квантованный"

        skimage.io.imshow(s, cmap="gray", ax=axes[i])
        axes[i].set_title(f"Разностный сигнал ({desc}) при e={e[i]}")
    plt.show()