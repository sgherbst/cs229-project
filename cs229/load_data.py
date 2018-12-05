import cv2
import numpy as np
import os.path
import os
import pickle
from glob import glob
from time import sleep
import sys
from math import degrees, pi
import imutils

from cs229.files import open_image, open_video, top_dir
from cs229.display import open_window, show_image, make_trackbar, get_trackbar, show_square_image
from cs229.image import img_to_mask
from cs229.util import FpsMon, TickTock
from cs229.annotation import Annotation
from cs229.orb import MyOrb

class ProcessedImg:
    def __init__(self, contour, patch, moments=None):
        # save settings
        self.contour = contour
        self.patch = patch

        # calculate image moments
        if moments is None:
            moments = cv2.moments(patch)

        self.moments = moments

    def contains_point(self, pt):
        return (cv2.pointPolygonTest(self.contour, tuple(pt), False) > 0)

    def rotate(self, angle):
        img = imutils.rotate_bound(self.patch, angle)
        return ProcessedImg(contour=None, patch=img)

    @staticmethod
    def make(img, contour):
        # create mask to extract just the part of the image in the contour
        mask = np.zeros(img.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, (255, 255, 255), -1)

        # set other parts of the image to zero
        masked = cv2.bitwise_and(img, img, mask=mask)

        # crop image to the region of interest
        xmin = np.min(contour[:, 0, 1])
        xmax = np.max(contour[:, 0, 1])
        ymin = np.min(contour[:, 0, 0])
        ymax = np.max(contour[:, 0, 0])
        cropped = masked[xmin:xmax + 1, ymin:ymax + 1]

        return ProcessedImg(contour=contour, patch=cropped)

def process(img):
    # define circular mask
    cmask = img_to_mask(img)

    # threshold image to find flies
    _, tmask = cv2.threshold(img, 115, 255, cv2.THRESH_BINARY_INV)

    # combine circular mask and fly mask
    fmask = cv2.bitwise_and(cmask, tmask)

    # find contours of flies
    _, contours, _ = cv2.findContours(fmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # sanity check for contours
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 200]
    if len(contours) == 0:
        print('Found no flies, skipping...')
        return
    if len(contours) == 1:
        print('Found only one fly, skipping...')
        return
    if len(contours) > 2:
        print('Found more than two flies, skipping...')
        return

    return [ProcessedImg.make(img, cnt) for cnt in contours]

def load_data(folder):
    X = []
    y = []

    for file in glob(os.path.join(folder, '*.json')):
        # load annotations
        anno = Annotation(file)

        if not anno.check_features(mh=1, ma=1, mj1=1, mj2=1,
                                   fh=1, fa=1, fj1=1, fj2=1):
            continue

        mj1 = anno.labels['mj1'][0]
        mj2 = anno.labels['mj2'][0]

        mh = anno.labels['mh'][0]
        ma = anno.labels['ma'][0]
        # note the order of y subtraction
        angle_male = np.arctan2(ma[1]-mh[1], mh[0]-ma[0])

        fj1 = anno.labels['fj1'][0]
        fj2 = anno.labels['fj2'][0]

        fh = anno.labels['fh'][0]
        fa = anno.labels['fa'][0]
        # note the order of y subtraction
        angle_female = np.arctan2(fa[1]-fh[1], fh[0]-fa[0])

        # load image
        img = cv2.imread(os.path.join(folder, anno.image_path), 0)
        res = process(img)

        for v in res:
            calc_angle = 0.5 * np.arctan(2 * v.moments['mu11'] / (v.moments['mu20'] - v.moments['mu02']))
            if v.moments['mu20'] < v.moments['mu02']:
                calc_angle += pi/2
            calc_angle = -calc_angle

            imutils.rotate_bound(v.patch, calc_angle)

            if v.contains_point(mj1) and v.contains_point(mj2):
                X.append(v.moments)
                y.append(0)
                should_display = True
                data_angle = angle_male
            elif v.contains_point(fj1) and v.contains_point(fj2):
                X.append(v.moments)
                y.append(1)
                should_display = True
                data_angle = angle_female

            if should_display:
                new = v.rotate(degrees(calc_angle))
                hu = cv2.HuMoments(new.moments).flatten()
                print(hu[6])
                show_square_image(new.patch)
                cv2.waitKey(2000)

    return X, y

def main():
    X, y = load_data(os.path.join(top_dir(), 'images', '12-01_10-41-12'))

if __name__ == '__main__':
    main()