import cv2 as cv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

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

img_name = '509'
img_path = f'{img_name}.jpg'

border = 20

img = cv.imread(img_path)
#template = cv.imread('template.jpg')
#template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

h, w = img.shape[:2]

img = cv.copyMakeBorder(img, border, border, border, border, cv.BORDER_CONSTANT, None, (255, 255, 255))
#plt.imshow(img[..., ::-1])
#plt.show()
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
plt.imshow(hsv)
plt.show()

h,s,v = cv.split(hsv)
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


ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
thr = cv.morphologyEx(thr, cv.MORPH_CLOSE, ker)

cnts, hierarchy = cv.findContours(thr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#gray = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
tmp_cnt = []
for i, cnt in enumerate(hierarchy[0]):
    if cnt[3] == -1:
        perimeter = cv.arcLength(cnts[i], True)
        tmp_cnt.append([i, perimeter])

idx = max(tmp_cnt, key=lambda c: c[1])[0]
epsilon = 0.1*cv.arcLength(cnts[idx],True)
approx = cv.approxPolyDP(cnts[idx],epsilon,True)
#cv.drawContours(img, [approx], 0, (0,255,0), 3)
#cv.drawContours(gray, cnts[idx], -1, (0,255,0), 3)
#plt.imshow(gray)
#plt.show()

if len(approx) != 4:
    print('Black board not found')

points = [ point[0] for point in approx ]

top_left, bottom_left, bottom_right, top_right = points[:4]

x, y = 0, 0
h, w = 0, 0

x1 = top_left[0] if top_left[0] > bottom_left[0] else bottom_left[0]
y1 = top_left[1] if top_left[1] < top_right[1] else top_right[1]

x2 = top_right[0] if top_right[0] < bottom_right[0] else bottom_right[0]
y2 = bottom_left[1] if bottom_left[1] < bottom_right[1] else bottom_right[1]



cropped = gray[y1:y2, x1:x2]
h, w = cropped.shape[:2]

offset_y = int(h//3.5)
offset_x = w//4

cropped = cropped[:h-offset_y, ...]
cropped = cropped[:, :w-offset_x]
cropped = cropped[:, offset_x:]
cropped_hsv = hsv[y1:y2, x1:x2]

#cropped[h-offset_y:, ...] = 0
#cropped[:, :offset_x] = 0
#cropped[:, w-offset_x:] = 0
#cropped_hsv[h-offset_y:, ...] = [0, 0, 0]

#blur = cv.boxFilter(cropped, -1, (31,31))



blur = cv.GaussianBlur(cropped, (31, 31), 60)
plt.imshow(blur, cmap='gray')
plt.show()

_, thr = cv.threshold(blur, 75, 255, cv.THRESH_BINARY)

plt.imshow(thr, cmap='gray')
plt.show()

thr = cv.bitwise_and(cropped, cropped, mask=thr)
canny = cv.Canny(thr, 30, 80, 5)
ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (21, 21))
#canny = cv.morphologyEx(canny, cv.MORPH_CLOSE, ker)
canny = cv.dilate(canny, ker)
plt.imshow(canny, cmap='gray')
plt.show()

"""
# ----------- White lateral band remove -----------------
thr = cv.bitwise_and(cropped_hsv, cropped_hsv, mask=thr)
plt.imshow(thr)
plt.show()

thr = cv.inRange(thr, (19, 15, 180), (24, 65, 230))
ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
thr = cv.morphologyEx(thr, cv.MORPH_CLOSE, ker)
thr = cv.morphologyEx(thr, cv.MORPH_OPEN, ker)
plt.imshow(thr, cmap='gray')
plt.show()

contours = cv.cvtColor(cropped, cv.COLOR_GRAY2BGR)

cnts, hierarchy = cv.findContours(thr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
tmp_cnt = []
for i, cnt in enumerate(hierarchy[0]):
    #if cnt[3] == -1:
    perimeter = cv.arcLength(cnts[i], True)
    tmp_cnt.append([i, perimeter])
    epsilon = 0.01 * perimeter
    approx = cv.approxPolyDP(cnts[i], epsilon, True)

    if len(approx) != 4:
        continue
    
    cv.drawContours(contours, [approx], 0, (0, 255, 0), 3)
    points = [ point[0] for point in approx ]
    
    top_left, bottom_left, bottom_right, top_right = points[:4]
        
#cv.drawContours(contours, cnts, -1, (0,0,255), 3)
plt.imshow(contours)
plt.show()

# ------------------------------------------------------
"""

"""
dst = cv.Canny(thr, 100, 255)
plt.imshow(dst, cmap='gray')
plt.show()

cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
cdstP = np.copy(cdst)

lines = cv.HoughLines(dst, 1, np.pi / 180, 800, None, 0, 0)

if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 5000*(-b)), int(y0 + 5000*(a)))
        pt2 = (int(x0 - 5000*(-b)), int(y0 - 5000*(a)))
        cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)

plt.imshow(cdst, cmap='gray')
plt.show()
"""