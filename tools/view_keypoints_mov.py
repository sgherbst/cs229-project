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
    cap, props = open_video('reduced')
    ok, img = cap.read()
    img = img[:, :, 0]
    img = cv2.resize(img, (512, 512))

    open_window()

    mask = img_to_mask(img)

    mon = FpsMon()

    kwargs = {
        'nfeatures': 500,
        'scaleFactor': 1.2,
        'nlevels': 3,
        'edgeThreshold': 6,
        'WTA_K': 2,
        'patchSize': 31
    }
    orb = cv2.ORB_create(**kwargs)

    while True:
        mon.tick()

        ok, img = cap.read()
        if not ok:
            break
        img = cv2.resize(img, (512, 512))

        # fmask = cv2.inRange(img, 1, 203)
        # img = cv2.bitwise_and(img, img, mask=fmask)

        cv2.setRNGSeed(0)

        kp, des = orb.detectAndCompute(img, mask=mask)
        out = cv2.drawKeypoints(img, kp, None)

        show_square_image(out)

        key = cv2.waitKey(props.t_ms)

        if key == ord('q'):
            break

    mon.done()

if __name__ == "__main__":
    main()