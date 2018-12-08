import os
import os.path
import cv2
import numpy as np
import matplotlib.pyplot as plt

from itertools import chain

from glob import glob

from cs229.files import top_dir
from cs229.annotation import Annotation
from cs229.image import img_to_mask
from cs229.contour import contour_label, find_contours
from cs229.patch import crop_to_contour

import joblib

CATEGORIES = ['mf', 'fm']
FEATURES = ['area_1', 'area_2', 'area_diff', 'dist_1', 'dist_2', 'ecc_1', 'ecc_2', 'wall_angle_1',
            'wall_angle_2']
X_JOBLIB_NAME = 'X_id.joblib'
Y_JOBLIB_NAME = 'y_id.joblib'

def wall_angle(cx_img, cy_img, cx_fly, cy_fly, angle_fly):
    angle_wall = np.arctan((cy_img-cy_fly)/(cx_fly-cx_img))
    rel_angle_fly = (angle_fly - angle_wall) % np.pi

    if rel_angle_fly < np.pi/2:
        return np.pi/2 - rel_angle_fly
    else:
        return rel_angle_fly - np.pi/2

def make_features(contour_1, contour_2, patch_1, patch_2, img):
    # create distance features
    cx_img = img.shape[1] // 2
    cy_img = img.shape[0] // 2

    cx_1, cy_1 = patch_1.estimate_center(absolute=True)
    cx_2, cy_2 = patch_2.estimate_center(absolute=True)

    dist_1 = np.hypot(cx_1-cx_img, cy_1-cy_img)
    dist_2 = np.hypot(cx_2-cx_img, cy_2-cy_img)

    # create eccentricity features
    ecc_1 = patch_1.estimate_eccentricity()
    ecc_2 = patch_2.estimate_eccentricity()

    # create area features
    area_1 = cv2.contourArea(contour_1)
    area_2 = cv2.contourArea(contour_2)
    area_diff = area_1 - area_2

    # create angle features
    wall_angle_1 = wall_angle(cx_img, cy_img, cx_1, cy_1, patch_1.estimate_angle())
    wall_angle_2 = wall_angle(cx_img, cy_img, cx_2, cy_2, patch_2.estimate_angle())

    # build feature vector
    features = []
    features.append(area_1)
    features.append(area_2)
    #features.append(area_diff)
    #features.append(dist_1)
    #features.append(dist_2)
    features.append(ecc_1)
    features.append(ecc_2)
    #features.append(wall_angle_1)
    #features.append(wall_angle_2)

    features = np.array(features, dtype=float)

    return features

def load_data():
    folders = ['12-04_17-54-43', '12-05-12-43-00', '12-07_16_45_00']
    #folders = ['12-07_16_45_00']
    folders = [os.path.join(top_dir(), 'images', folder) for folder in folders]
    folders = [glob(os.path.join(folder, '*.json')) for folder in folders]

    files = chain(*folders)

    X = []
    y = []

    for f in files:
        anno = Annotation(f)
        img = cv2.imread(anno.image_path, 0)

        mask = img_to_mask(img)
        contours = find_contours(img, mask=mask)

        male = None
        female = None
        ok = True

        # sift through contours to find one corresponding to male and the other to female
        for contour in contours:
            label = contour_label(anno, contour)
            if label == 'male':
                if male is None:
                    male = contour
                else:
                    ok = False
                    anno.warn('Found two males, skipping...')
                    break
            elif label == 'female':
                if female is None:
                    female = contour
                else:
                    ok = False
                    anno.warn('Found two females, skipping...')
                    break

        if (male is not None) and (female is not None) and ok:
            male_patch = crop_to_contour(img, male)
            female_patch = crop_to_contour(img, female)

            X.append(make_features(male, female, male_patch, female_patch, img))
            y.append(CATEGORIES.index('mf'))

            X.append(make_features(female, male, female_patch, male_patch, img))
            y.append(CATEGORIES.index('fm'))

    # assemble features
    X = np.array(X, dtype=float)

    # assemble labels
    y = np.array(y, dtype=int)

    return X, y

def main():
    X, y = load_data()

    joblib.dump(X, X_JOBLIB_NAME)
    joblib.dump(y, Y_JOBLIB_NAME)

if __name__ == '__main__':
    main()