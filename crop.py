import cv2 as cv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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
#plt.imshow(thr, cmap='gray')
#plt.show()
"""
blur = cv.GaussianBlur(hsv, (11,11), 20)
thr = cv.inRange(hsv, (9, 12, 120), (14, 20, 140))
thr2 = cv.inRange(hsv, (6, 15, 120), (32, 165, 220))
thr = cv.bitwise_or(thr, thr2)
#ret3,thr = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
plt.imshow(thr, cmap='gray')
plt.show()
plt.imshow(thr2, cmap='gray')
plt.show()
#plt.imshow(thr, cmap='gray')
#plt.show()

#plt.imshow(cv.medianBlur(thr, 21))

ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
thr = cv.morphologyEx(thr, cv.MORPH_OPEN, ker)
plt.imshow(thr, cmap='gray')
plt.show()

ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
thr = cv.morphologyEx(thr, cv.MORPH_CLOSE, ker)
plt.imshow(thr, cmap='gray')
plt.show()
"""

#dx = cv.Sobel()
ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
thr = cv.morphologyEx(thr, cv.MORPH_CLOSE, ker)
#plt.imshow(thr, cmap='gray')
#plt.show()

cnts, hierarchy = cv.findContours(thr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
#print(hierarchy[:10])
tmp_cnt = []
for i, cnt in enumerate(hierarchy[0]):
    if cnt[3] == -1:
        perimeter = cv.arcLength(cnts[i], True)
        tmp_cnt.append([i, perimeter])

idx = max(tmp_cnt, key=lambda c: c[1])[0]
print(idx)
epsilon = 0.1*cv.arcLength(cnts[idx],True)
approx = cv.approxPolyDP(cnts[idx],epsilon,True)
cv.drawContours(gray, [approx], 0, (0,255,0), 3)
#cv.drawContours(gray, cnts[idx], -1, (0,255,0), 3)
#plt.imshow(gray)
#plt.show()

if len(approx) != 4:
    print('Black board not found')

points = [ point[0] for point in approx ]

#upper_most = min(points, key=lambda p: p[1])
points.sort(key=lambda p: p[1])
bottom_most = points[0]
upper_most = points[-1]
print(bottom_most)
print(upper_most)

x, y = upper_most
#w, h = 
plt.imshow(img[upper_most, bottom_most])
plt.show()