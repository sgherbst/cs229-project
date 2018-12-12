import cv2
import numpy as np

def bound_point(pt, img):
    def clip(dim):
        return np.clip(np.round(pt[dim]), 0, img.shape[1-dim] - 1).astype(int)

    return (clip(0), clip(1))

def img_to_mask(img):
    rows, cols = img.shape[0], img.shape[1]

    # sanity checks
    assert rows==cols

    # compute radius
    r = (rows-1)//2

    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (r, r), r, 255, -1)

    return mask