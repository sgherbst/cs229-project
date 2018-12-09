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
    make_trackbar('nfeatures', 10, 100)
    make_trackbar('nOctaveLayers', 1, 10)
    make_trackbar('contrastThreshold', 2, 10)
    make_trackbar('edgeThreshold', 29, 40)
    make_trackbar('sigma', 20, 100)
    make_trackbar('showKeypoints', 1, 1)
    make_trackbar('blur', 5, 10)
    make_trackbar('downsamp', 1, 4)

    old_test = None
    old_nfeatures = None
    old_nOctaveLayers = None
    old_contrastThreshold = None
    old_edgeThreshold = None
    old_sigma = None
    old_blur = None
    old_downsamp = None

    refresh_image = False
    refresh_SIFT = False

    img = None
    sift = None
    mask = None
    kp = None

    while True:
        test = get_trackbar('test')
        nfeatures = get_trackbar('nfeatures', min=1)
        nOctaveLayers = get_trackbar('nOctaveLayers', min=1)
        contrastThreshold = get_trackbar('contrastThreshold', min=1)*0.01
        edgeThreshold = get_trackbar('edgeThreshold', min=1)
        sigma = get_trackbar('sigma', min=1)*0.01
        downsamp = get_trackbar('downsamp', min=1)

        if (old_test != test) or (old_downsamp != downsamp):
            refresh_image = True
        if (old_nfeatures != nfeatures or old_nOctaveLayers != nOctaveLayers or old_contrastThreshold != contrastThreshold or
            old_edgeThreshold != edgeThreshold or old_sigma != sigma):
            refresh_SIFT = True

        old_test = test
        old_nfeatures = nfeatures
        old_nOctaveLayers = nOctaveLayers
        old_contrastThreshold = contrastThreshold
        old_edgeThreshold = edgeThreshold
        old_sigma = sigma
        old_downsamp = downsamp

        if refresh_SIFT:
            kwargs = dict(nfeatures=nfeatures, nOctaveLayers=nOctaveLayers, contrastThreshold=contrastThreshold,
                          edgeThreshold=edgeThreshold, sigma=sigma)
            print('Reloading SIFT...')
            print(kwargs)
            sift =cv2.xfeatures2d.SIFT_create(**kwargs)

        if refresh_image:
            print('Reloading image...')
            file_path = os.path.join(top_dir(), 'images', 'sift_test', '{}.bmp'.format(test))
            img = cv2.imread(file_path, 0)
            img = img[::downsamp, ::downsamp]
            # blurred = cv2.GaussianBlur(img,(blur,blur),0)
            # unsharp = (1+amount)*img.astype(float) - amount*blurred.astype(float)
            # img = np.clip(np.round(unsharp), 0, 255).astype(np.uint8)

        if refresh_image or refresh_SIFT:
            cv2.setRNGSeed(0)
            if img is not None and sift is not None:
                tick = perf_counter()
                kp, des = sift.detectAndCompute(img, mask=mask)
                tock = perf_counter()
                print('SIFT took {:0.1f} ms'.format(1e3*(tock-tick)))

            refresh_image = False
            refresh_SIFT = False

        if img is not None:
            out = cv2.resize(img, (downsamp * img.shape[1], downsamp * img.shape[0]))
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
            if get_trackbar('showKeypoints') == 1 and kp is not None:
                cv2.setRNGSeed(0)
                for k in kp:
                    cv2.circle(out, (int(downsamp*k.pt[0]), int(downsamp*k.pt[1])), 3, (0, 0, 255), thickness=-1)
            show_image(out)

        key = cv2.waitKey(25)

        if key == ord('q'):
            break

if __name__ == "__main__":
    main()