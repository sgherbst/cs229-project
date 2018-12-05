import cv2
import numpy as np
import os.path
import os
import pickle
import matplotlib.pyplot as plt
from math import hypot
from time import perf_counter

from cs229.files import open_image, open_video, top_dir
from cs229.display import open_window, show_image, make_trackbar, get_trackbar, show_square_image
from cs229.image import img_to_mask
from cs229.util import FpsMon, TickTock
from cs229.annotation import Annotation
from cs229.orb import MyOrb

def get_image(n, prefix='test'):
    print('Getting image {}.'.format(n))
    
    folder = os.path.join(top_dir(), 'images', 'handpicked')
    img_path = os.path.join(folder, '{}{}.png'.format(prefix, n))

    try:
        img = cv2.imread(img_path, 0)
        mask = img_to_mask(img)
    except:
        img = None
        mask = None

    return img, mask

def recenter(v, side, size):
    if v-side < 0:
        v = side
    elif v+side >= size:
        v = size-side-1

    assert (0 <= v-side) and (v+side < size)

    return v

def disp_patch(img, i, j, patch_size=127, disp_size=512):
    rows, cols = img.shape

    half_size = patch_size//2
    i = recenter(i, half_size, rows)
    j = recenter(j, half_size, cols)

    patch = img[i-half_size:i+half_size, j-half_size:j+half_size]
    patch = patch.copy()
    patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    cv2.circle(patch,(half_size,half_size), 2, (0,0,255), -1)
    cv2.imshow('detail', cv2.resize(patch, (disp_size, disp_size)))

def nn_argmin(x, y, kp):
    return min(range(len(kp)), key=lambda i:hypot(kp[i].pt[0]-x, kp[i].pt[1]-y))

def l2(x, y):
    return np.linalg.norm(x-y)

# def preprocess(img):
#     # kernel = np.array((
#     #     [0, 1, 0],
#     #     [1, -4, 1],
#     #     [0, 1, 0]), dtype="int")
#     #
#     # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
#     # out = cv2.filter2D(img, -1, kernel)
#
#     out = 255 - img
#
#     return out

def sharpen(img):
    blur = cv2.GaussianBlur(img, (7,7), 0)
    zeroed = img.astype(float) - blur.astype(float)
    scale = 3*np.std(zeroed)
    print(np.min(zeroed), np.max(zeroed), scale)

    out = 128*(zeroed/scale + 1)
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out.astype(np.uint8)

def preprocess(img, mask):
    new_mask = (img < 127)*(mask==255)

    out = cv2.GaussianBlur(img, (3,3), 0)*new_mask

    return out

def main(downsamp=1):
    mouse_data = open_window(log_clicks=True)

    make_trackbar('test', 1, 9)
    test = None
    old_test = None

    orb = MyOrb()

    kp = None
    des = None
    pt0 = None
    pt1 = None

    while True:
        cv2.setRNGSeed(0)

        test = get_trackbar('test', min=1)

        if test != old_test:
            print('Reloading image...')
            file_path = os.path.join(top_dir(), 'images', 'handpicked', 'test{}.png'.format(test))
            img = cv2.imread(file_path, 0)
            img = cv2.resize(img, (512, 512))
            mask = img_to_mask(img)

            tick = perf_counter()
            orb.detect(img, mask=mask)
            kp = orb.kp
            tock = perf_counter()
            print('SIFT detect: {} ms'.format(1e3 * (tock - tick)))

            tick = perf_counter()
            des = orb.compute(img, kp)
            tock = perf_counter()
            print('SIFT compute: {} ms'.format(1e3 * (tock - tick)))

        old_test = test

        out = img.copy()
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        if len(mouse_data.clicks)==2:
            # get click location
            print('Finding new point...')
            clicks = mouse_data.clicks
            x0, y0 = clicks[0].x * downsamp, clicks[0].y * downsamp
            x1, y1 = clicks[1].x * downsamp, clicks[1].y * downsamp

            mouse_data.clicks = []

            # find index of closest keypoint
            pt0 = nn_argmin(x0, y0, kp)
            pt1 = nn_argmin(x1, y1, kp)

            # display image patch around that keypoint
            # disp_patch(img1, int(kp1[query_idx].pt[1]), int(kp1[query_idx].pt[0]))

            # find most similar keypoint in the second image
            print('Feature distance: {}'.format(l2(des[pt0, :], des[pt1, :])))

        if pt0 is not None and pt1 is not None:
            cv2.circle(out, (int(kp[pt0].pt[0]), int(kp[pt0].pt[1])), 3, (0, 0, 255), -1)
            cv2.circle(out, (int(kp[pt1].pt[0]), int(kp[pt1].pt[1])), 3, (0, 255, 0), -1)

        show_image(out[::downsamp, ::downsamp])

        key = cv2.waitKey(25)

        if key == ord('q'):
            break

if __name__ == "__main__":
    main()