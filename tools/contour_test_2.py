import cv2
import numpy as np
import os.path
import os
import pickle
import matplotlib.pyplot as plt
from math import hypot
from time import perf_counter
import timeit

from cs229.files import open_image, open_video, top_dir
from cs229.display import open_window, show_image, make_trackbar, get_trackbar, show_square_image
from cs229.image import img_to_mask
from cs229.util import FpsMon, TickTock
from cs229.annotation import Annotation
from cs229.orb import MyOrb

from skimage.feature import hog as skhog

def main():
    mouse_data = open_window(log_clicks=True)

    test = None
    old_test = None

    img = None

    cap, props = open_video('cropped')
    ok, img = cap.read()
    img = img[:, :, 0]
    img = cv2.resize(img, (512, 512))
    dim = img.shape[0]
    cmask = img_to_mask(img)

    while True:
        ok, img = cap.read()
        if not ok:
            break
        img = img[:, :, 0]

        tick = perf_counter()
        img = cv2.resize(img, (512, 512))
        tock = perf_counter()

        _, tmask = cv2.threshold(img, 115, 255, cv2.THRESH_BINARY_INV)
        fmask = cv2.bitwise_and(cmask, tmask)

        im2, contours, hierarchy = cv2.findContours(fmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        contours = sorted(contours, key=lambda cnt: -cv2.contourArea(cnt))
        contour = contours[0]

        omask = np.zeros((dim, dim), dtype=np.uint8)
        cv2.drawContours(omask, [contour], 0, (255, 255, 255), -1)


        xmin = np.min(contour[:, 0, 1])
        xmax = np.max(contour[:, 0, 1])
        ymin = np.min(contour[:, 0, 0])
        ymax = np.max(contour[:, 0, 0])

        out = cv2.bitwise_and(img, img, mask=omask)
        out = out[xmin:xmax+1, ymin:ymax+1]

        M = cv2.moments(out)


        print('Processing time: {} ms'.format((tock-tick)*1e3))

        show_square_image(out)

        key = cv2.waitKey(25)

        if key == ord('q'):
            break


if __name__ == "__main__":
    main()