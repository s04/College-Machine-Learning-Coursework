# SAUL O'DRISCOLL 17333932
import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def apply_kernel(image, kernel):
    m, n = kernel.shape
    if (m == n):
        y, x = image.shape
        y = y - m + 1
        x = x - m + 1
        new_image = np.zeros((y,x))
        for i in range(y):
            for j in range(x):
                new_image[i][j] = np.sum(image[i:i+m, j:j+m]*kernel)
                
    return new_image

def convo(image, kern):
    imgOut = image
    if (kern.shape[0]==kern.shape[1]):
        imgOut = apply_kernel(image,kern)
    return imgOut

test_files = ["android.jpg", "index.jpeg"]

for pic in test_files:
    im = np.asarray(Image.open(pic))
    gray = rgb2gray(im)

    kernel1 = np.array([[-1, -1, -1], [-1, -8, -1], [-1, -1, -1]])
    kernel2 = np.array([[0, -1, 0], [-1, -8, -1], [0, -1, 0]])

    print("Convolution 1")
    kernel1_result = convo(gray,kernel1)
    kernel2_result = convo(gray,kernel2)

    i = 2
    while(i <= 6):
        print("Convolution {0}".format(i))
        kernel1_result = convo(kernel1_result,kernel1)
        kernel2_result = convo(kernel2_result,kernel2)
        i += 1

    f, axarr = plt.subplots(2,3)

    k1y, k1x = kernel1_result.shape
    k2y, k2x = kernel2_result.shape

    zoom = 2.2

    axarr[0,0].set_title("Original")
    axarr[0,0].axis('off')
    axarr[0,0].imshow(im)
    axarr[1,0].set_title("Original")
    axarr[1,0].axis('off')
    axarr[1,0].imshow(im)
    
    axarr[0,1].set_title("kernel1")
    axarr[0,1].axis('off')
    axarr[0,1].imshow(kernel1, cmap="gray")
    axarr[1,1].set_title("kernel2")
    axarr[1,1].axis('off')
    axarr[1,1].imshow(kernel2, cmap="gray")

    axarr[0,2].set_title("kernel1 zoomed output")
    axarr[0,2].axis('off')
    axarr[0,2].imshow(kernel1_result[:math.floor(k1x/zoom),:math.floor(k1y/zoom)], cmap="gray")
    axarr[1,2].set_title("kernel2 zoomed output")
    axarr[1,2].axis('off')
    axarr[1,2].imshow(kernel2_result[:math.floor(k2x/zoom),:math.floor(k2y/zoom)], cmap="gray")

    plt.show()
    plt.close()