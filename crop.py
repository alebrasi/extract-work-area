import cv2 as cv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

img_name = '509R'
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
gray = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
tmp_cnt = []
for i, cnt in enumerate(hierarchy[0]):
    if cnt[3] == -1:
        perimeter = cv.arcLength(cnts[i], True)
        tmp_cnt.append([i, perimeter])

idx = max(tmp_cnt, key=lambda c: c[1])[0]
epsilon = 0.1*cv.arcLength(cnts[idx],True)
approx = cv.approxPolyDP(cnts[idx],epsilon,True)
#cv.drawContours(gray, [approx], 0, (0,255,0), 3)
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

cropped = img[y1:y2, x1:x2]

