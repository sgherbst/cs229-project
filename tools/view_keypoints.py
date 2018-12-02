import cv2
import numpy as np
import os.path
import os
import pickle
from math import hypot

from cs229.files import open_image
from cs229.display import open_window, show_square_image, make_trackbar, get_trackbar, show_image
from cs229.image import img_to_mask
from cs229.util import FpsMon

def main(size=512):
    img = open_image()
    img = cv2.resize(img, (size, size))

    mouse_data = open_window(log_clicks=True)
    show_square_image(img)

    make_trackbar('test', 1, 8)
    make_trackbar('nfeatures', 500, 1000)
    make_trackbar('scaleFactor', 120, 200)
    make_trackbar('nlevels', 3, 16)
    make_trackbar('edgeThreshold', 6, 64)
    make_trackbar('WTA_K', 2, 4)
    make_trackbar('patchSize', 31, 64)
    make_trackbar('showKeypoints', 1, 1)
    make_trackbar('dist', 1, 25)

    mask = img_to_mask(img)

    mon = FpsMon()
    mousePos = None

    kwargs = {
        'nfeatures': get_trackbar('nfeatures', min=1),
        'scaleFactor': get_trackbar('scaleFactor', min=1.1, fixed_point=0.01),
        'nlevels': get_trackbar('nlevels', min=1),
        'edgeThreshold': get_trackbar('edgeThreshold', min=1),
        'WTA_K': get_trackbar('WTA_K', min=2),
        'patchSize': get_trackbar('patchSize', min=1)
    }
    # print(kwargs)
    orb = cv2.ORB_create(**kwargs)

    while True:
        mon.tick()

        print(mouse_data.clicks)
        if mouse_data.clicks:
            mousePos = mouse_data.clicks[-1]

        img = open_image('test{}'.format(get_trackbar('test', min=1)))
        img = cv2.resize(img, (size, size))

        # fmask = cv2.inRange(img, 1, 203)
        # img = cv2.bitwise_and(img, img, mask=fmask)

        cv2.setRNGSeed(0)

        kp = orb.detect(img, mask=mask)
        if mousePos is not None:
            dist = get_trackbar('dist', min=1)
            kp = min(kp, key=lambda v: hypot(v.pt[0]-mousePos.x, v.pt[1]-mousePos.y))
            print(hypot(kp.pt[0] - mousePos.x, kp.pt[1] - mousePos.y))
            kp = [kp]

        out = img.copy()
        if get_trackbar('showKeypoints') == 1:
            out = cv2.drawKeypoints(img, kp, None)

        show_image(out)

        key = cv2.waitKey(20)

        if key == ord('q'):
            break
        elif key == ord('d'):
            mousePos = None

    mon.done()

if __name__ == "__main__":
    main()