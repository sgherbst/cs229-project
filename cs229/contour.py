import cv2
import os
import os.path
import matplotlib.pyplot as plt
import numpy as np
from time import perf_counter

from cs229.files import top_dir
from cs229.image import img_to_mask

DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))

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

def find_body_contours(img, mask=None, canny_thresh_1=32, canny_thresh_2=40, canny_sobel=3):
    # detect edges
    canny = cv2.Canny(img, canny_thresh_1, canny_thresh_2, canny_sobel)

    # apply dilation to fix edges
    dilated = cv2.dilate(canny, DILATE_KERNEL, iterations=1)

    # mask off the outside
    if mask is not None:
        dilated = cv2.bitwise_and(dilated, mask)

    # get contours
    _, contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return result
    return contours

def find_core_contours(img, mask=None, fly_color='black', thresh_int=115):
    # determine thresholding type
    if fly_color.lower() == 'black':
        thresh_type = cv2.THRESH_BINARY_INV
    elif fly_color.lower() == 'white':
        thresh_type = cv2.THRESH_BINARY
    else:
        raise Exception('Invalid fly color.')

    # threshold image
    _, bw = cv2.threshold(img, thresh_int, 255, thresh_type)

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
    contours = find_core_contours(img, mask=mask)

    print(len(contours))

    plt.imshow(img)
    plt.show()

if __name__ =='__main__':
    main()