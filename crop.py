import cv2 as cv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

# TODO: Fare una funzione più corta
# https://gist.github.com/flashlib/e8261539915426866ae910d55a3f9959
def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # if use Euclidean distance, it will run in error when the object
    # is trapezoid. So we should use the same simple y-coordinates order method.

    # now, sort the right-most coordinates according to their
    # y-coordinates so we can grab the top-right and bottom-right
    # points, respectively
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return [tl, tr, br, bl]

def remove_lateral_white_bands(img):
    band_range = [(19, 15, 180), (24, 65, 230)]
    #band_range_clahe = [(25, 18, 170), (35, 90, 235)]

    img = cv.GaussianBlur(img, (31, 31), 60)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # Remove noise
    hsv = cv.pyrDown(hsv)
    hsv = cv.pyrUp(hsv)

    thr = cv.inRange(hsv, band_range[0], band_range[1])
    ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (50, 50))
    #thr = cv.morphologyEx(thr, cv.MORPH_CLOSE, ker)
    thr = cv.dilate(thr, ker)

    #contours = cv.cvtColor(cropped, cv.COLOR_GRAY2BGR)
    
    rects = []

    cnts, hierarchy = cv.findContours(thr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for i, cnt in enumerate(hierarchy[0]):

        # Contour approximation
        perimeter = cv.arcLength(cnts[i], True)
        epsilon = 0.02 * perimeter
        approx = cv.approxPolyDP(cnts[i], epsilon, True)

        # Discard non squares/rectangles and non convex        
        if len(approx) != 4 and not cv.isContourConvex(cnts[i]):
            continue
        
        points = [ point[0] for point in approx ]
        points = order_points(np.array(points))
        
        top_left, top_right, bottom_right, bottom_left = points[:4]

        norm = lambda pt1, pt2: math.sqrt(((pt1[0]-pt2[0])**2) + ((pt1[1]-pt2[1])**2))

        w, h = norm(top_left, top_right), norm(top_left, bottom_left)
        if h < 1750 or w > h:
            continue
  
        rects.append(cv.minAreaRect(approx))

    h_i, w_i = img.shape[:2]
    mask = np.ones((h_i, w_i), dtype='uint8')
    c_x = h_i//2
    for rect in rects:
        rect = np.int0(cv.boxPoints(rect))

        cv.drawContours(mask, [rect], 0, 0, -1)
    
    return mask

matplotlib.use('TkAgg')

img_name = '509'
img_path = f'{img_name}.jpg'

border = 20

img = cv.imread(img_path)

h, w = img.shape[:2]

img = cv.copyMakeBorder(img, border, border, border, border, cv.BORDER_CONSTANT, None, (255, 255, 255))
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

h,s,v = cv.split(hsv)
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
points = order_points(np.array(points))

top_left, top_right, bottom_right, bottom_left = points[:4]

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

cropped_bgr = img[y1:y2, x1:x2]
cropped_bgr = cropped_bgr[:h-offset_y, ...]
cropped_bgr = cropped_bgr[:, :w-offset_x]
cropped_bgr = cropped_bgr[:, offset_x:]


blur = cv.GaussianBlur(cropped, (31, 31), 60)
#plt.imshow(blur, cmap='gray')
#plt.show()

_, mask = cv.threshold(blur, 75, 255, cv.THRESH_BINARY)
ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 25))
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, ker)

masked = cv.bitwise_and(cropped_bgr, cropped_bgr, mask=mask)

mask = remove_lateral_white_bands(masked)

print(masked.shape)
print(mask.shape)

masked = cv.bitwise_and(masked, masked, mask=mask)
plt.imshow(masked)
plt.show()

"""
# ----- Morphological gradient -------
gradient_ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
gradient = cv.morphologyEx(mask, cv.MORPH_GRADIENT, gradient_ker)
plt.imshow(gradient, cmap='gray')
plt.show()
# -------------------------------------
"""

"""
cropped = cv.pyrDown(cropped)
cropped = cv.pyrUp(cropped)

dst = cv.Canny(cropped, 90, 160)
plt.imshow(dst, cmap='gray')
plt.show()

cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
#cdst = masked

lines = cv.HoughLines(dst, 1, np.pi / 180, 200, None, 0, 0)
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