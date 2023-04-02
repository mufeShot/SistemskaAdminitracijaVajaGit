import numpy as np
import cv2
from matplotlib import pyplot as plt

def my_roberts(slika):
    #vaša implementacija
    return slika_robov

def my_prewitt(slika):
    #vaša implementacija
    return slika_robov

def my_sobel(slika):
    #vaša implementacija
    return slika_robov

def canny(slika, sp_prag, zg_prag):
    #vaša implementacija
    return slika_robov 

def main():
    # read image
    img = cv2.imread('lenna.png')

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow(img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
