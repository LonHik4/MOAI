import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from scipy.signal import convolve2d


img = cv.imread('04_boat.tif', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
def grad():
    Hx = np.array([[-1, 1]])
    Hy = np.array([[-1, 1]]).transpose()
    Gx = convolve2d(img, Hx, boundary="symm", mode="same")
    Gy = convolve2d(img, Hy, boundary="symm", mode="same")
    Grad = np.sqrt((Gx**2) + (Gy**2))

    plt.subplot(231)
    plt.title("Исходное изображение")
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)

    plt.subplot(232)
    plt.title("Оценка градиента")
    plt.imshow(Grad, cmap='gray', vmin=0, vmax=255)

    plt.subplot(233)
    plt.title("Гистограмма оценки градиента")
    plt.hist(Grad.ravel(), bins='auto')

    plt.subplot(234)
    Gx = np.ceil((Gx-20)/255)*255
    plt.title("Частная производная по X")
    plt.imshow(Gx, cmap='gray', vmin=0, vmax=255)

    plt.subplot(235)
    Gy = np.ceil((Gy-20)/255)*255
    plt.title("Частная производная по Y")
    plt.imshow(Gy, cmap='gray', vmin=0, vmax=255)

    plt.subplot(236)
    G = np.ceil((Grad-25)/255)*255
    plt.title("Контуры градиентный метод")
    plt.imshow(G, cmap='gray', vmin=0, vmax=255)
    plt.show()



def laplasian():
    mask1 = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    mask2 = np.array([[1,0,1],[0,-4,0],[1,0,1]])/2
    mask3 = np.array([[1,1,1],[1,-8,1],[1,1,1]])/3
    laplas1 = np.abs(convolve2d(img, mask1, boundary="symm", mode="same"))
    laplas2 = np.abs(convolve2d(img, mask2, boundary="symm", mode="same"))
    laplas3 = np.abs(convolve2d(img, mask3, boundary="symm", mode="same"))
    contur1 = np.ceil((laplas1 - 30) / 255) * 255
    contur2 = np.ceil((laplas2 - 30) / 255) * 255
    contur3 = np.ceil((laplas3 - 30) / 255) * 255

    plt.subplot(341)
    plt.title("Исходное изображение")
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)

    plt.subplot(342)
    plt.title("Маска 1")
    plt.imshow(laplas1, cmap='gray', vmin=0, vmax=255)

    plt.subplot(343)
    plt.title("Гистограмма оценки лапласиана (маска 1)")
    plt.hist(laplas1.ravel(), bins='auto')

    plt.subplot(344)
    plt.title("Контуры метода лапласиана (маска 1)")
    plt.imshow(contur1, cmap='gray', vmin=0, vmax=255)

    plt.subplot(345)
    plt.title("Исходное изображение")
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)

    plt.subplot(346)
    plt.title("Маска 2")
    plt.imshow(laplas2, cmap='gray', vmin=0, vmax=255)

    plt.subplot(347)
    plt.title("Гистограмма оценки лапласиана (маска 2)")
    plt.hist(laplas2.ravel(), bins='auto')

    plt.subplot(348)
    plt.title("Контуры метода лапласиана (маска 2)")
    plt.imshow(contur2, cmap='gray', vmin=0, vmax=255)

    plt.subplot(349)
    plt.title("Исходное изображение")
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)

    plt.subplot(3,4,10)
    plt.title("Маска 3")
    plt.imshow(laplas3, cmap='gray', vmin=0, vmax=255)

    plt.subplot(3,4,11)
    plt.title("Гистограмма оценки лапласиана (маска 3)")
    plt.hist(laplas3.ravel(), bins='auto')

    plt.subplot(3,4,12)
    plt.title("Контуры метода лапласиана (маска 3)")
    plt.imshow(contur3, cmap='gray', vmin=0, vmax=255)
    plt.show()

def pruit():
    mask1 = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])/6
    mask2 = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])/6
    s1 = convolve2d(img, mask1, boundary="symm", mode="same")
    s2 = convolve2d(img, mask2, boundary="symm", mode="same")
    Grad = np.sqrt((s1**2) + (s2**2))
    s1 = np.ceil((s1-5)/255)*255
    s2 = np.ceil((s2 - 5) / 255) * 255
    contur = np.ceil((Grad-15)/255)*255

    plt.subplot(231)
    plt.title("Исходное изображение")
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)

    plt.subplot(232)
    plt.title("Оценка метода оператора Прюитт")
    plt.imshow(Grad, cmap='gray', vmin=0, vmax=255)

    plt.subplot(233)
    plt.title("Гистограмма оценки Прюитта")
    plt.hist(Grad.ravel(), bins='auto')

    plt.subplot(234)
    plt.title("Частная производная по первому направлению")
    plt.imshow(s1, cmap='gray', vmin=0, vmax=255)

    plt.subplot(235)
    plt.title("Частная производная по второму направлению")
    plt.imshow(s2, cmap='gray', vmin=0, vmax=255)

    plt.subplot(236)
    plt.title("Контуры метод Прюитта")
    plt.imshow(contur, cmap='gray', vmin=0, vmax=255)
    plt.show()

def agreement_laplassian():
    mask = np.array([[2, -1, 2], [-1, -4, -1], [2, -1, 2]]) / 3
    laplas = np.abs(convolve2d(img, mask, boundary="symm", mode="same"))
    contur = np.ceil((laplas - 25) / 255) * 255

    plt.subplot(221)
    plt.title("Исходное изображение")
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)

    plt.subplot(222)
    plt.title("Оценка согласования для лапласиана")
    plt.imshow(laplas, cmap='gray', vmin=0, vmax=255)

    plt.subplot(223)
    plt.title("Гистограмма оценки согласования")
    plt.hist(laplas.ravel(), bins='auto')

    plt.subplot(224)
    plt.title("Контуры согласования")
    plt.imshow(contur, cmap='gray', vmin=0, vmax=255)
    plt.show()

if __name__ == "__main__":
    #grad()
    #laplasian()
    #pruit()
    agreement_laplassian()