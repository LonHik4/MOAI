import matplotlib
import numpy as np
from skimage.util import random_noise
from scipy.signal import convolve2d
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def chessboard():
    chess = np.array([160, 96])
    board = (np.indices((128, 128)) // 16).sum(axis=0) % 2
    return chess[board]


def noise_calc(img, snr, mode=False, p=0.0):
    noise_img = img.copy()
    disp = np.var(img) / 255 ** 2
    if not mode:
        noise_img = random_noise(img/255, var=disp/snr) * 255
        noise = noise_img - img + 128
    else:
        noise = random_noise(img, mode="s&p", amount=p) * 255
        noise_img[noise == 0] = 0
        noise_img[noise == 255] = 255
        print("a")

    return noise_img, noise, np.var(noise_img), np.var(noise)


def filter_median(img):
    return median_filter(img, size=3)


def filter_linear(img):
    mask = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    return convolve2d(img, mask, boundary='symm', mode='same')


def apply_filters(img):
    return filter_median(img), filter_linear(img)


def calc_kc(orig_img, noised_img, filtered_img):
    e1 = np.power((filtered_img - orig_img), 2).mean()
    e2 = np.power((noised_img - orig_img), 2).mean()
    kc = e1/e2
    return kc


def draw_image(img, plt_num, title):
    plt.subplot(plt_num)
    plt.title(title)
    plt.axis('off')
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)


def print_info(title, original_img, noised_img, filtered_median, filtered_linear):
    print(f"{title}")
    print(f"дисперсия ошибок фильтрации для медианого фильтра: {np.var(filtered_median - original_img)}")
    print(f"дисперсия ошибок фильтрации для линейного фильтра: {np.var(filtered_linear - original_img)}")
    print(f"коэф подавления шума для медианого: фильтра {calc_kc(original_img, noised_img, filtered_median)}")
    print(f"коэф подавления шума для линейного фильтра: {calc_kc(original_img, noised_img, filtered_linear)}")


if __name__ == '__main__':
    chessbrd = chessboard()
    image_disp = np.var(chessbrd)
    draw_image(chessbrd, 111, f"Original image (Var = {image_disp})")
    plt.show()

    # Task 1
    additive_noise_img, additive_noise, noised_disp, noise_disp = noise_calc(chessbrd, 1, False, 0.1)
    median, linear = apply_filters(additive_noise_img)
    draw_image(additive_noise, 221, f"Noise (Var = {noise_disp})")
    draw_image(additive_noise_img, 222, f"Image + White noise (Var = {noised_disp})")
    draw_image(median, 223, "Median filter")
    draw_image(linear, 224, "Linear filter")
    print_info("Task 1, snr = 1", chessbrd, additive_noise_img, median, linear)
    plt.show()

    additive_noise_img, additive_noise, noised_disp, noise_disp = noise_calc(chessbrd, 10, False, 0.1)
    median, linear = apply_filters(additive_noise_img)
    draw_image(additive_noise, 221, f"Noise (Var = {noise_disp})")
    draw_image(additive_noise_img, 222, f"Image + White noise (Var = {noised_disp})")
    draw_image(median, 223, "Median filter")
    draw_image(linear, 224, "Linear filter")
    print_info("Task 1, snr = 10", chessbrd, additive_noise_img, median, linear)
    plt.show()

    # Task 2
    impulse_noise_img, impulse_noise, noised_disp, noise_disp = noise_calc(chessbrd, 1, True, 0.1)
    median, linear = apply_filters(impulse_noise_img)
    draw_image(impulse_noise, 221, f"Noise (Var = {noise_disp})")
    draw_image(impulse_noise_img, 222, f"Image + White noise (Var = {noised_disp})")
    draw_image(median, 223, "Median filter")
    draw_image(linear, 224, "Linear filter")
    print_info("Task 2, p = 0.1", chessbrd, impulse_noise_img, median, linear)
    plt.show()

    impulse_noise_img, impulse_noise, noised_disp, noise_disp = noise_calc(chessbrd, 1, True, 0.3)
    median, linear = apply_filters(impulse_noise_img)
    draw_image(impulse_noise, 221, f"Noise (Var = {noise_disp})")
    draw_image(impulse_noise_img, 222, f"Image + White noise (Var = {noised_disp})")
    draw_image(median, 223, "Median filter")
    draw_image(linear, 224, "Linear filter")
    print_info("Task 2, p = 0.3", chessbrd, impulse_noise_img, median, linear)
    plt.show()
