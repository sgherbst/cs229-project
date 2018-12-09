import cv2
import cv2.xfeatures2d
import os.path
import os
import numpy as np
from time import perf_counter

from cs229.files import top_dir
from cs229.display import open_window, make_trackbar, get_trackbar, show_image


def adjust_gamma(image, gamma=1.0):
    # from: https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/

    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def main():
    open_window()

    make_trackbar('test', 0, 10)
    make_trackbar('hessianThreshold', 40, 100)
    make_trackbar('nOctaves', 4, 10)
    make_trackbar('nOctaveLayers', 3, 10)
    make_trackbar('showKeypoints', 1, 1)

    old_test = None
    old_hessianThreshold = None
    old_nOctaves = None
    old_nOctaveLayers = None

    refresh_image = False
    refresh_SURF = False

    img = None
    sift = None
    mask = None
    kp = None

    while True:
        test = get_trackbar('test')
        hessianThreshold = get_trackbar('hessianThreshold', min=1)*10
        nOctaves = get_trackbar('nOctaves', min=1)
        nOctaveLayers = get_trackbar('nOctaveLayers', min=1)

        if (old_test != test):
            refresh_image = True
        if (old_hessianThreshold != hessianThreshold or old_nOctaves != nOctaves or old_nOctaveLayers != nOctaveLayers):
            refresh_SURF = True

        old_test = test
        old_hessianThreshold = hessianThreshold
        old_nOctaves = nOctaves
        old_nOctaveLayers = nOctaveLayers

        if refresh_SURF:
            kwargs = dict(hessianThreshold=hessianThreshold, nOctaves=nOctaves, nOctaveLayers=nOctaveLayers)
            print('Reloading SIFT...')
            print(kwargs)
            sift =cv2.xfeatures2d.SURF_create(**kwargs)

        if refresh_image:
            print('Reloading image...')
            file_path = os.path.join(top_dir(), 'images', 'sift_test', '{}.bmp'.format(test))
            img = cv2.imread(file_path, 0)

        if refresh_image or refresh_SURF:
            cv2.setRNGSeed(0)
            if img is not None and sift is not None:
                tick = perf_counter()
                kp, des = sift.detectAndCompute(img, mask=mask)
                tock = perf_counter()
                print('SURF took {:0.1f} ms'.format(1e3*(tock-tick)))

            refresh_image = False
            refresh_SURF = False

        if img is not None:
            out = img.copy()
            if get_trackbar('showKeypoints') == 1 and kp is not None:
                cv2.setRNGSeed(0)
                out = cv2.drawKeypoints(img, kp, None)
            show_image(out)

        key = cv2.waitKey(25)

        if key == ord('q'):
            break

if __name__ == "__main__":
    main()