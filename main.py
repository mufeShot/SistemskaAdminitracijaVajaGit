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
