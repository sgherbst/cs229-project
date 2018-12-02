import cv2
import numpy as np
import os.path
import os
import pickle

from cs229.files import open_image, open_video
from cs229.display import open_window, show_square_image, make_trackbar, get_trackbar
from cs229.image import img_to_mask
from cs229.util import FpsMon

def main():
    #img = open_image()
    cap, props = open_video('test1')
    ok, img = cap.read()
    img = img[:, :, 0]

    img = cv2.resize(img, (512, 512))

    open_window()
    show_square_image(img)

    make_trackbar('test', 1, 5)
    make_trackbar('max_thresh', 203, 255)
    make_trackbar('min_thresh', 0, 255)

    cmask = img_to_mask(img)

    mon = FpsMon()

    while True:
        mon.tick()

        #img = open_image('test{}'.format(get_trackbar('test', min=1)))
        ok, img = cap.read()
        if not ok:
            break
        img = img[:, :, 0]

        img = cv2.resize(img, (512, 512))

        fmask = cv2.inRange(img, get_trackbar('min_thresh', min=1), get_trackbar('max_thresh', min=1))
        print(get_trackbar('max_thresh', min=1))
        #fmask = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        # out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))

        #_, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        mask = cv2.bitwise_and(cmask, fmask)
        out = cv2.bitwise_and(img, img, mask=mask)

        show_square_image(out)
        key = cv2.waitKey(props.t_ms)
        if key == ord('q'):
            break

    mon.done()

if __name__ == "__main__":
    main()