import numpy as np
import cv2
from matplotlib import pyplot as plt

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
