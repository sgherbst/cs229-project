import cv2
import os
import os.path
import matplotlib.pyplot as plt
import numpy as np
from time import perf_counter

from cs229.files import top_dir
from cs229.image import img_to_mask

ERODE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

def in_contour(pt, contour):
    return cv2.pointPolygonTest(contour, tuple(pt), False) > 0

def contour_label(anno, contour):
    contains_m = in_contour(anno.get('mp')[0], contour)
    contains_f = in_contour(anno.get('fp')[0], contour)

    if contains_m:
        if contains_f:
            return 'both'
        else:
            return 'male'
    else:
        if contains_f:
            return 'female'
        else:
            return 'neither'

def find_contours(img, mask=None, thresh='low', fly_color='black'):
    if thresh == 'low':
        thresh_int = 115
        erode = False
    elif thresh == 'high':
        thresh_int = 210
        erode = False
    else:
        raise Exception('Invalid threshold type.')

    # determine thresholding type
    if fly_color.lower() == 'black':
        thresh_type = cv2.THRESH_BINARY_INV
    elif fly_color.lower() == 'white':
        thresh_type = cv2.THRESH_BINARY
    else:
        raise Exception('Invalid fly color.')

    # threshold image
    _, bw = cv2.threshold(img, thresh_int, 255, thresh_type)

    # apply erosion if needed
    if erode:
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, ERODE_KERNEL)

    # apply mask to image if desired
    if mask is not None:
        bw = cv2.bitwise_and(bw, mask)

    # extract contours
    _, contours, _ = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # return contours
    return contours

def main():
    image_path = os.path.join(top_dir(), 'images', '12-01_10-41-12', '2.bmp')
    img = cv2.imread(image_path, 0)

    mask = img_to_mask(img)
    contours = find_contours(img, mask=mask)

    print(len(contours))

    plt.imshow(img)
    plt.show()

if __name__ =='__main__':
    main()