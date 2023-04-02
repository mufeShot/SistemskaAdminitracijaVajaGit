import numpy as np
import cv2
from matplotlib import pyplot as plt

def convolve(image, kernel):
    # Get the dimensions of the kernel and the image
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate the amount of padding needed
    pad_height = (kernel_height - 1) // 2
    pad_width = (kernel_width - 1) // 2

    # Create a zero-padded image
    padded_image = np.zeros((image_height + 2 * pad_height, image_width + 2 * pad_width))
    padded_image[pad_height:-pad_height, pad_width:-pad_width] = image


    # Create an output array to hold the convolved image
    output = np.zeros((image_height, image_width))

    # Loop over each pixel in the image
    for i in range(pad_height, image_height + pad_height):
        for j in range(pad_width, image_width + pad_width):
            # Extract the patch from the padded image
            patch = padded_image[i - pad_height:i + pad_height + 1, j - pad_width:j + pad_width + 1]

            # Apply the kernel to the patch
            output[i - pad_height, j - pad_width] = np.sum(patch * kernel)

    return output

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

def my_prewitt(image):
    # convert image to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # remove this line

    # apply prewitt kernel to x and y directions
    prewitt_kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    prewitt_x = convolve(image, prewitt_kernel_x)
    prewitt_y = convolve(image, prewitt_kernel_y)

    # combine x and y edges
    prewitt_edges = np.sqrt(np.square(prewitt_x) + np.square(prewitt_y))
    prewitt_edges = (prewitt_edges / prewitt_edges.max()) * 255

    # convert edges to uint8
    prewitt_edges = prewitt_edges.astype(np.uint8)

    return prewitt_edges

def my_sobel(gray):

    # define sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # pad image
    #padded_image = np.pad(gray, pad_width=1, mode='constant', constant_values=0)

    # apply sobel kernels using convolve function
    edges_x = convolve(gray, sobel_x)
    edges_y = convolve(gray, sobel_y)

    # calculate magnitude of edges
    magnitude = np.sqrt(edges_x ** 2 + edges_y ** 2)

    # normalize magnitude to 0-255 range
    magnitude *= 255.0 / np.max(magnitude)

    return magnitude.astype(np.uint8)

def canny(slika, sp_prag, zg_prag):
    blurred = cv2.GaussianBlur(slika, (5, 5), 0)

    return cv2.Canny(blurred, sp_prag, zg_prag)

def canny_without_gauss(slika, sp_prag, zg_prag):
    return cv2.Canny(slika, sp_prag, zg_prag)

def spremeni_kontrast(slika, alfa, beta):
    return cv2.convertScaleAbs(slika, alpha=alfa, beta=beta)

def main():
    # read image
    img = cv2.imread('lenna.png')

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Originalna slika', img)
    roberts_edges = my_roberts(gray)
    cv2.imshow('Roberts1', roberts_edges)
    slika2 = spremeni_kontrast(gray, -3, 2)
    cv2.imshow('Roberts2', my_roberts(slika2))
    slika3 = spremeni_kontrast(gray, 2, 10)
    cv2.imshow('Roberts3', my_roberts(slika3))
    slika4 = spremeni_kontrast(gray, 1, 40)
    cv2.imshow('Roberts4', my_roberts(slika4))
    slika5 = spremeni_kontrast(gray, 4, 75)
    cv2.imshow('Roberts5', my_roberts(slika5))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
