import cv2
import numpy as np
import os.path
import os
import pickle
import matplotlib.pyplot as plt
from math import hypot

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

def hamming(x, y):
    x = np.unpackbits(x.astype(np.uint8))
    y = np.unpackbits(y.astype(np.uint8))

    n = x.shape[0]
    return n - x.dot(y) - (1-x).dot(1-y)

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

def main(downsamp=3):
    mouse_data = open_window(log_clicks=True)

    orb = MyOrb()

    img1, mask1 = get_image(1)
    img2, mask2 = get_image(9)

    img1 = preprocess(img1, mask1)
    img2 = preprocess(img2, mask2)

    orb.detect(img1, mask=mask1)
    kp1 = orb.kp
    des1 = orb.compute(img1, kp1)

    orb.detect(img2, mask=mask2)
    kp2 = orb.kp
    des2 = orb.compute(img2, kp2)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw the matches one at a time.
    count = 0
    matches = []

    while True:
        cv2.setRNGSeed(0)

        if mouse_data.clicks:
            print('Finding new point...')
            last_click = mouse_data.clicks[-1]
            mouse_data.clicks = []
            x = last_click.x*downsamp
            y = last_click.y*downsamp
            query_idx = nn_argmin(x, y, kp1)
            disp_patch(img1, int(kp1[query_idx].pt[1]), int(kp1[query_idx].pt[0]))
            train_idx = min(range(len(kp2)), key=lambda i: hamming(des1[query_idx, :], des2[i, :]))
            distance = hamming(des1[query_idx, :], des2[train_idx, :])
            matches = [cv2.DMatch(query_idx, train_idx, distance)]

        img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
        # print(matches[count].imgIdx, matches[count].queryIdx, matches[count].trainIdx,
        #       matches[count].distance, kp1[matches[count].queryIdx].pt[0], kp1[matches[count].queryIdx].pt[1],
        #       kp2[matches[count].trainIdx].pt[0], kp2[matches[count].trainIdx].pt[1])
        # match = matches[count]
        # idx1 = match.queryIdx
        # idx2 = match.trainIdx
        #
        # print(hamming(des1[idx1, :], des2[idx2, :]), match.distance)

        show_image(img3[::downsamp, ::downsamp])

        key = cv2.waitKey(25)

        if key == ord('q'):
            break
        elif key == ord('m'):
            count = min(len(matches), count+1)
        elif key == ord('n'):
            count = max(0, count-1)

if __name__ == "__main__":
    main()