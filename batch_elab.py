import cv2 as cv
from crop3 import crop_img
import errno
import os
from tqdm import tqdm


in_path = '/home/alebrasi/Documents/tesi/Dataset/sessioni_crop/cropped/'
out_path = '/home/alebrasi/Documents/tesi/Dataset/sessioni_crop/resize_512'

l = len(in_path)
for (dirpath, dirnames, filenames) in os.walk(in_path):
    p = dirpath[l:]
    if p == "":
        continue

    out_dir = f'{out_path}/{p}'
    os.makedirs(out_dir, exist_ok=True)
    print(dirpath)
    
    for file in tqdm(filenames):
        file_path = f'{dirpath}/{file}' 
        img = cv.imread(file_path)
        try:
            out_file_path = f'{out_dir}/{file}'
            cropped = crop_img(img, 512, 0, pre_cropped=True, hough_thr=200, band_height_thr=1800)
            cv.imwrite(out_file_path, cropped)
        except:
            print(f'Error in: {file_path}')