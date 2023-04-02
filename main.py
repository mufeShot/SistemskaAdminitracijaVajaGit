import numpy as np
import cv2
from matplotlib import pyplot as plt

def my_roberts(image):
    kernel_x = np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]])
    kernel_y = np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]])

    # Pad the image with zeros to handle edges
    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='edge')

    # Compute gradients in x and y directions
    grad_x = convolve(padded_image, kernel_x)[1:-1, 1:-1]
    grad_y = convolve(padded_image, kernel_y)[1:-1, 1:-1]

    # Compute magnitude of gradient
    gradient_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Normalize gradient magnitude
    gradient_mag = (gradient_mag - np.min(gradient_mag)) / (np.max(gradient_mag) - np.min(gradient_mag)) * 255

    # Convert gradient magnitude to uint8
    gradient_mag = gradient_mag.astype(np.uint8)

    return gradient_mag

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
