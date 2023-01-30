import cv2 as cv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import time

from utils import order_points, segment_by_angle_kmeans, segmented_intersections

def remove_lateral_white_bands(img):
    norm = lambda pt1, pt2: math.sqrt(((pt1[0]-pt2[0])**2) + ((pt1[1]-pt2[1])**2))

    band_range = [(19, 15, 180), (24, 65, 230)]

    img = cv.GaussianBlur(img, (31, 31), 60)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # ------------------- Remove noise -------------
    hsv = cv.pyrDown(hsv)
    hsv = cv.pyrUp(hsv)
    # ----------------------------------------------

    # --------------- Lateral white bands thresholding ---------
    thr = cv.inRange(hsv, band_range[0], band_range[1])
    ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (50, 50))
    thr = cv.dilate(thr, ker)
    # ----------------------------------------------------------

    rects = []

    cnts, hierarchy = cv.findContours(thr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for i, cnt in enumerate(hierarchy[0]):

        # -------------------- Contour approximation --------------
        perimeter = cv.arcLength(cnts[i], True)
        epsilon = 0.02 * perimeter
        approx = cv.approxPolyDP(cnts[i], epsilon, True)
        # ---------------------------------------------------------

        # Discard non squares/rectangles and non convex polygon
        if len(approx) != 4 and not cv.isContourConvex(cnts[i]):
            continue
        
        points = [ point[0] for point in approx ]
        points = order_points(np.array(points))
        
        top_left, top_right, bottom_right, bottom_left = points[:4]

        # Filtering false positive rectangles
        w, h = norm(top_left, top_right), norm(top_left, bottom_left)
        if h < 1750 or w > h:
            continue
  
        rects.append(cv.minAreaRect(approx))

    # ------------------------ Mask creation without bands --------
    h_i, w_i = img.shape[:2]
    mask = np.ones((h_i, w_i), dtype='uint8')
    c_x = h_i//2
    for rect in rects:
        rect = np.int0(cv.boxPoints(rect))
        cv.drawContours(mask, [rect], 0, 0, -1)
    
    return mask

def crop_img(img, img_resize_size=512, resize_offset=0, pre_cropped=False):
    # --------------- Black board thresholding and cropping -----------------
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thr = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV)
    plt.imshow(gray, cmap='gray')
    plt.show()
    plt.imshow(thr, cmap='gray')
    plt.show()

    ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
    thr = cv.morphologyEx(thr, cv.MORPH_CLOSE, ker)

    cnts, hierarchy = cv.findContours(thr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    tmp_cnt = []
    for i, cnt in enumerate(hierarchy[0]):
        if cnt[3] == -1:
            perimeter = cv.arcLength(cnts[i], True)
            tmp_cnt.append([i, perimeter])

    idx = max(tmp_cnt, key=lambda c: c[1])[0]
    epsilon = 0.1*cv.arcLength(cnts[idx],True)
    approx = cv.approxPolyDP(cnts[idx],epsilon,True)

    if len(approx) != 4:
        print('Black board not found')

    points = [ point[0] for point in approx ]
    points = order_points(np.array(points))

    top_left, top_right, bottom_right, bottom_left = points[:4]

    x1 = top_left[0] if top_left[0] > bottom_left[0] else bottom_left[0]
    y1 = top_left[1] if top_left[1] < top_right[1] else top_right[1]

    x2 = top_right[0] if top_right[0] < bottom_right[0] else bottom_right[0]
    y2 = bottom_left[1] if bottom_left[1] < bottom_right[1] else bottom_right[1]

    cropped_bgr = img[y1:y2, x1:x2]
    h, w = cropped_bgr.shape[:2]

    offset_y = int(h//3.5)
    offset_x = w//4

    cropped_bgr = cropped_bgr[:h-offset_y, ...]
    cropped_bgr = cropped_bgr[:, :w-offset_x]
    cropped_bgr = cropped_bgr[:, offset_x:]

    # ------------------------------------------------------------------------------

    # ----------------- Mask extraction ----------------------------------------
    gray = cv.cvtColor(cropped_bgr, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (31, 31), 60)

    _, back_mask = cv.threshold(blur, 75, 255, cv.THRESH_BINARY)
    ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 25))
    back_mask = cv.morphologyEx(back_mask, cv.MORPH_CLOSE, ker)

    band_mask = remove_lateral_white_bands(cropped_bgr)

    mask = back_mask & band_mask

    ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (50, 50))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, ker)
    # ---------------------------------------------------------------------

    # --------------- Morphological gradient ----------------------------
    gradient_ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
    gradient = cv.morphologyEx(mask, cv.MORPH_GRADIENT, gradient_ker)
    # -----------------------------------------------------------------

    # ------------- Intersections ----------------------------
    #https://stackoverflow.com/questions/46565975/find-intersection-point-of-two-lines-drawn-using-houghlines-opencv
    lines = cv.HoughLines(gradient, 1, np.pi / 180, 200, None, 0, 0)
    
    segmented = segment_by_angle_kmeans(lines)
    intersections = segmented_intersections(segmented)
    points = np.array([ pt[0] for pt in intersections ])
    for pt in intersections:
        pt = pt[0]
        x, y = pt
    # ---------------------------------------------------------

    # ------- Extract bounding box and fix orientation --------
    rect = cv.minAreaRect(points)
    box = cv.boxPoints(rect)
    center, _, angle = rect
    box = np.int0(box)

    #cdst = cropped_bgr

    if angle < 50:
        h, w = cropped_bgr.shape[:2]
        rot_m = cv.getRotationMatrix2D(center, angle, 1.0)
        cdst = cv.warpAffine(cropped_bgr, rot_m, (w, h))

    # ----------------------------------------------------------

    # --------- Resize ----------------------------------
    min_y = min(box, key=lambda p: p[1])[1] + resize_offset
    max_y = max(box, key=lambda p: p[1])[1] - resize_offset
    min_x = min(box, key=lambda p: p[0])[0] + resize_offset
    max_x = max(box, key=lambda p: p[0])[0] - resize_offset

    cdst = cdst[min_y:max_y, min_x:max_x, ...]
    cdst = cv.resize(cdst, (img_resize_size, img_resize_size))
    # ---------------------------------------

    return cdst

if __name__ == '__main__':
    matplotlib.use('TkAgg')

    img_name = '508'
    img_path = f'{img_name}.jpg'
    border = 20
    resize_offset = 0
    img_size = 512
    img = cv.imread(img_path)
    cropped = crop_img(img, 1024, 0, pre_cropped=True)
    plt.imshow(cropped)
    plt.show()
    #cv.imwrite('cropped.jpg', cropped)