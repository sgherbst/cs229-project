import cv2
import cv2.xfeatures2d
import numpy as np
import os.path
import os
import pickle
from math import hypot

from cs229.files import open_image, top_dir
from cs229.display import open_window, show_square_image, make_trackbar, get_trackbar, show_image
from cs229.image import img_to_mask
from cs229.util import FpsMon

# def preprocess(img):
#     blur = cv2.GaussianBlur(img, (31,31), 0)
#     zeroed = img.astype(float) - blur.astype(float)
#     scale = 3*np.std(zeroed)
#     print(np.min(zeroed), np.max(zeroed), scale)
#
#     out = 128*(zeroed/scale + 1)
#     out = np.clip(out, 0, 255)
#
#     return out.astype(np.uint8)

def main(disp_size=1000):
    open_window()

    make_trackbar('test', 1, 9)
    make_trackbar('nfeatures', 30, 50)
    make_trackbar('nOctaveLayers', 5, 10)
    make_trackbar('contrastThreshold', 3, 10)
    make_trackbar('edgeThreshold', 9, 40)
    make_trackbar('sigma', 16, 25)
    make_trackbar('showKeypoints', 1, 1)

    old_test = None
    old_nfeatures = None
    old_nOctaveLayers = None
    old_contrastThreshold = None
    old_edgeThreshold = None
    old_sigma = None

    refresh_image = False
    refresh_SIFT = False

    img = None
    sift = None
    mask = None
    kp = None

    while True:
        test = get_trackbar('test', min=1)
        nfeatures = get_trackbar('nfeatures', min=1)*10
        nOctaveLayers = get_trackbar('nOctaveLayers', min=1)
        contrastThreshold = get_trackbar('contrastThreshold', min=1)*0.01
        edgeThreshold = get_trackbar('edgeThreshold', min=1)
        sigma = get_trackbar('sigma', min=1)*0.1

        if old_test != test:
            refresh_image = True
        if (old_nfeatures != nfeatures or old_nOctaveLayers != nOctaveLayers or old_contrastThreshold != contrastThreshold or
            old_edgeThreshold != edgeThreshold or old_sigma != sigma
        ):
            refresh_SIFT = True

        old_test = test
        old_nfeatures = nfeatures
        old_nOctaveLayers = nOctaveLayers
        old_contrastThreshold = contrastThreshold
        old_edgeThreshold = edgeThreshold
        old_sigma = sigma

        if refresh_SIFT:
            kwargs = dict(nfeatures=nfeatures, nOctaveLayers=nOctaveLayers, contrastThreshold=contrastThreshold,
                          edgeThreshold=edgeThreshold, sigma=sigma)
            print('Reloading SIFT...')
            print(kwargs)
            sift =cv2.xfeatures2d.SIFT_create(**kwargs)

        if refresh_image:
            print('Reloading image...')
            file_path = os.path.join(top_dir(), 'images', 'handpicked', 'test{}.png'.format(test))
            img = cv2.imread(file_path, 0)
            mask = img_to_mask(img)
            #img = preprocess(img)

        if refresh_image or refresh_SIFT:
            cv2.setRNGSeed(0)
            if img is not None and sift is not None:
                kp = sift.detect(img, mask=mask)

            refresh_image = False
            refresh_SIFT = False

        if img is not None:
            out = img.copy()
            if get_trackbar('showKeypoints') == 1 and kp is not None:
                cv2.setRNGSeed(0)
                out = cv2.drawKeypoints(img, kp, None)
            show_square_image(out)

        key = cv2.waitKey(25)

        if key == ord('q'):
            break

if __name__ == "__main__":
    main()