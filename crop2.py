import cv2 as cv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def create_gaborfilter():
    # This function is designed to produce a set of GaborFilters 
    # an even distribution of theta values equally distributed amongst pi rad / 180 degree

    filters = []
    num_filters = 16
    ksize = 35  # The local area to evaluate
    sigma = 2.0  # Larger Values produce more edges
    lambd = 20.0
    gamma = 0.5
    psi = 0  # Offset value - lower generates cleaner results
    for theta in np.arange(0, np.pi, np.pi / num_filters):  # Theta is the orientation for edge detection
        kern = cv.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv.CV_64F)
        kern /= 1.0 * kern.sum()  # Brightness normalization
        filters.append(kern)
    return filters


def apply_filter(img, filters):
# This general function is designed to apply filters to our image
    # First create a numpy array the same size as our input image
    newimage = np.zeros_like(img)

    # Starting with a blank image, we loop through the images and apply our Gabor Filter
    # On each iteration, we take the highest value (super impose), until we have the max value across all filters
    # The final image is returned
    depth = -1 # remain depth same as original image

    for kern in filters:  # Loop through the kernels in our GaborFilter
        image_filter = cv.filter2D(img, depth, kern)  #Apply filter to image

        # Using Numpy.maximum to compare our filter and cumulative image, taking the higher value (max)
        np.maximum(newimage, image_filter, newimage)
    return newimage

matplotlib.use('TkAgg')

img_name = '11'
img_path = f'{img_name}.jpg'

border = 20

img = cv.imread(img_path)
template = cv.imread('template.png')
template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

h, w = img.shape[:2]

img = cv.copyMakeBorder(img, border, border, border, border, cv.BORDER_CONSTANT, None, (255, 255, 255))
#plt.imshow(img[..., ::-1])
#plt.show()
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

h,s,v = cv.split(hsv)

clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(30,30))
v = clahe.apply(v)
plt.imshow(v, cmap='gray')
plt.show()

filters = create_gaborfilter()
s = apply_filter(s, filters)
v = apply_filter(v, filters)
plt.imshow(v, cmap='gray')
plt.show()
plt.imshow(v, cmap='gray')
plt.show()

hsv = cv.merge([h,s,v])
rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
plt.imshow(rgb[..., ::-1])
plt.show()
"""
plt.imshow(s)
plt.show()
plt.imshow(v)
plt.show()
"""
#plt.imshow(hsv)
#plt.show()
black_thr = [(0, 0, 0), (255, 255, 100)]
white_thr = [(0, 0, 120), (255, 255, 255)]
thr = cv.inRange(hsv, black_thr[0], black_thr[1])
"""
dx = cv.Sobel(v, -1, 1, 0)
dy = cv.Sobel(v, 0, 1)
plt.imshow(dx)
plt.imshow(dy)
"""
"""
ker = cv.getStructuringElement(cv.MORPH_RECT, (600, 10))
morphed = cv.morphologyEx(v, cv.MORPH_OPEN, ker)
plt.imshow(img)
plt.show()
plt.imshow(morphed, cmap='gray')
plt.show()
ker2 = cv.getStructuringElement(cv.MORPH_RECT, (10, 200))
morphed = cv.morphologyEx(v, cv.MORPH_OPEN, ker2)
plt.imshow(morphed, cmap='gray')
plt.show()
"""