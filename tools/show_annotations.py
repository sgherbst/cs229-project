import cv2
import numpy as np
import os.path
import os
import pickle

from cs229.files import open_image, open_video, top_dir
from cs229.display import open_window, show_image, make_trackbar, get_trackbar
from cs229.image import img_to_mask
from cs229.util import FpsMon, TickTock
from cs229.annotation import Annotation
from cs229.orb import MyOrb

def get_image(n):
    folder = os.path.join(top_dir(), 'images', '12-01_10-41-12')
    img_path = os.path.join(folder, '{}.bmp'.format(n))
    json_path = os.path.join(folder, '{}.json'.format(n))

    try:
        img = cv2.imread(img_path, 0)
        mask = img_to_mask(img)
    except:
        img = None
        mask = None

    try:
        anno = Annotation(json_path)
    except:
        anno = None

    return img, mask, anno

def main():
    count = 1
    reload = True

    img, mask, anno = None, None, None
    open_window()

    make_trackbar('feature', 0, 4)
    make_trackbar('thresh', 6, 10)

    orb = MyOrb()
    mon_detect = TickTock('DETECT')
    mon_compute = TickTock('COMPUTE')

    while True:
        if reload:
            img, mask, anno = get_image(count)
            reload = False

        if img is not None and mask is not None:
            cv2.setRNGSeed(0)
            mon_detect.tick()
            orb.detect(img, mask=mask)
            mon_detect.tock()

            if anno is not None:
                kp, _ = orb.split_kp(anno.num_to_lis(get_trackbar('feature')), thresh=get_trackbar('thresh'))
            else:
                kp = orb.kp

            out = img
            if kp:
                out = cv2.drawKeypoints(out, kp, None)

            show_image(out)

        key = cv2.waitKey(25)

        if key == ord('q'):
            break
        elif key == ord('m'):
            count += 1
            reload = True
        elif key == ord('n'):
            count -= 1
            reload = True

if __name__ == "__main__":
    main()