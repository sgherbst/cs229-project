import cv2
import numpy as np
import os.path
import os
import pickle
from math import hypot

from cs229.files import open_image, top_dir
from cs229.display import open_window, show_square_image, make_trackbar, get_trackbar, show_image
from cs229.image import img_to_mask
from cs229.util import FpsMon

def preprocess(img):
    blur = cv2.GaussianBlur(img, (31,31), 0)
    zeroed = img.astype(float) - blur.astype(float)
    scale = 3*np.std(zeroed)
    print(np.min(zeroed), np.max(zeroed), scale)

    out = 128*(zeroed/scale + 1)
    out = np.clip(out, 0, 255)

    return out.astype(np.uint8)

def main(disp_size=1000):
    open_window()

    make_trackbar('test', 1, 9)
    make_trackbar('nfeatures', 5, 10)
    make_trackbar('scaleFactor', 12, 20)
    make_trackbar('nlevels', 3, 16)
    make_trackbar('edgeThreshold', 6, 64)
    make_trackbar('WTA_K', 2, 4)
    make_trackbar('patchSize', 31, 64)
    make_trackbar('showKeypoints', 1, 1)

    old_test = None
    old_nfeatures = None
    old_scaleFactor = None
    old_nlevels = None
    old_edgeThreshold = None
    old_WTA_K = None
    old_patchSize = None

    refresh_image = False
    refresh_ORB = False

    img = None
    orb = None

    while True:
        test = get_trackbar('test', min=1)
        nfeatures = get_trackbar('nfeatures', min=1)*100
        scaleFactor = get_trackbar('scaleFactor', min=1.1, fixed_point=0.1)
        nlevels = get_trackbar('nlevels', min=1)
        edgeThreshold = get_trackbar('edgeThreshold', min=1)
        WTA_K = get_trackbar('WTA_K', min=2)
        patchSize = get_trackbar('patchSize', min=1)

        if old_test != test:
            refresh_image = True
        if (old_nfeatures != nfeatures or old_scaleFactor != scaleFactor or old_nlevels != nlevels or
            old_edgeThreshold != edgeThreshold or old_WTA_K != WTA_K or old_patchSize != patchSize
        ):
            refresh_ORB = True

        old_test = test
        old_nfeatures = nfeatures
        old_scaleFactor = scaleFactor
        old_nlevels = nlevels
        old_edgeThreshold = edgeThreshold
        old_WTA_K = WTA_K
        old_patchSize = patchSize

        if refresh_image:
            print('Reloading image...')
            file_path = os.path.join(top_dir(), 'images', 'handpicked', 'test{}.png'.format(test))
            img = cv2.imread(file_path, 0)
            mask = img_to_mask(img)
            #img = preprocess(img)
            refresh_image = False

        if refresh_ORB:
            print('Reloading ORB...')
            orb =cv2.ORB_create(nfeatures=nfeatures, scaleFactor=scaleFactor, nlevels=nlevels,
                                edgeThreshold=edgeThreshold, WTA_K=WTA_K, patchSize=patchSize)
            refresh_ORB = False


        cv2.setRNGSeed(0)
        kp = orb.detect(img, mask=mask)

        out = img.copy()
        if get_trackbar('showKeypoints') == 1:
            out = cv2.drawKeypoints(img, kp, None)

        show_square_image(out)

        key = cv2.waitKey(25)

        if key == ord('q'):
            break

if __name__ == "__main__":
    main()