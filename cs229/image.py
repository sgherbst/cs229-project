import cv2
import numpy as np

def crop_to_circle(frame, roi_circle):
    (cx, cy), cr = roi_circle
    return frame[cy-cr:cy+cr,cx-cr:cx+cr]

def circle_to_mask(roi_circle):
    (cx, cy), cr = roi_circle
    width = 2 * cx + 1
    height = 2*cy + 1

    mask = np.zeros((height, width), np.uint8)
    cv2.circle(mask, *roi_circle, 255, -1)

    return mask

def img_to_mask(img):
    rows, cols = img.shape

    # sanity checks
    assert rows==cols

    # compute radius
    r = (rows-1)//2

    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (r, r), r, 255, -1)

    return mask