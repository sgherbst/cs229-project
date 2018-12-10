import cv2
import os
import os.path
import matplotlib.pyplot as plt
import numpy as np
from time import perf_counter

from cs229.files import top_dir
from cs229.image import img_to_mask

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))

def in_contour(pt, contour):
    return cv2.pointPolygonTest(contour, tuple(pt), False) > 0

def erode(img):
    return cv2.erode(img, KERNEL, iterations=1)

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

def find_wing_contours(img, mask=None, fly_color='black'):
    # threshold image
    bw = threshold_wings(img=img, mask=mask, fly_color=fly_color)

    # extract contours
    _, contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return contours
    return contours

def threshold_wings(img, mask=None, fly_color='black', mean_offset=5):
    # determine thresholding type
    if fly_color == 'black':
        thresh_type = cv2.THRESH_BINARY_INV
    elif fly_color == 'white':
        thresh_type = cv2.THRESH_BINARY
    else:
        raise Exception('Invalid fly color.')

    # apply median filter
    blurred = cv2.medianBlur(img, 5)
    mean_in_roi = np.sum((mask==255)*img)/np.sum(mask==255)

    # threshold image a bit below the median
    _, bw = cv2.threshold(blurred, mean_in_roi-mean_offset, 255, thresh_type)

    # erode image
    eroded = erode(bw)

    # mask to ROI
    masked = cv2.bitwise_and(eroded, mask)

    return masked

def threshold_core(img, mask=None, fly_color='black', thresh_int=115):
    # determine thresholding type
    if fly_color == 'black':
        thresh_type = cv2.THRESH_BINARY_INV
    elif fly_color == 'white':
        thresh_type = cv2.THRESH_BINARY
    else:
        raise Exception('Invalid fly color.')

    # create the black and white image
    _, bw = cv2.threshold(img, thresh_int, 255, thresh_type)

    # apply mask to thresholded image
    masked = cv2.bitwise_and(bw, mask)

    # return the thresholded and masked image
    return masked

def find_core_contours(img, mask=None, fly_color='black'):
    # threshold image
    bw = threshold_core(img=img, mask=mask, fly_color=fly_color)

    # extract contours
    _, contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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