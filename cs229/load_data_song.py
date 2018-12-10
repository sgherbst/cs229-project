DEBUG = True

import numpy as np
import cv2
import matplotlib.pyplot as plt

from cs229.files import open_video
from cs229.image import img_to_mask
from cs229.contour import find_core_contours
from cs229.patch import crop_to_contour
from cs229.train_is_fly import IsFlyPredictor
from cs229.train_id import IdPredictor
from cs229.train_orientation import PosePredictor
from cs229.load_data_wing import male_fly_patch
from cs229.train_wing import WingPredictor

import joblib

FEATURES = ['right', 'left']
X_JOBLIB_NAME = 'X_song.joblib'

def load_data():
    X = []

    # prepare video
    cap, props = open_video('cropped')
    _, img = cap.read()
    img = img[:, :, 0]
    mask = img_to_mask(img)

    # load predictors
    is_fly_predictor = IsFlyPredictor()
    id_predictor = IdPredictor()
    pose_predictor = PosePredictor('male')
    wing_predictor = WingPredictor()

    frame_count = 0

    while True:
        # read frame
        ok, img = cap.read()

        if not ok:
            break

        frame_count += 1
        if frame_count % 100 == 0:
            print('Frame #{}'.format(frame_count))

        # placeholder for data
        data = [None for _ in range(len(FEATURES))]

        # make grayscale
        img = img[:, :, 0]

        # extract contours
        contours = find_core_contours(img, mask=mask)

        # find flies
        flies = []
        for contour in contours:
            if is_fly_predictor.predict(contour) == 'one':
                flies.append(contour)

        # find male
        if len(flies) != 2:
            X.append(data)
            continue

        contour_1 = flies[0]
        contour_2 = flies[1]

        patch_1 = crop_to_contour(img, contour_1)
        patch_2 = crop_to_contour(img, contour_2)

        label = id_predictor.predict(contour_1, contour_2, patch_1, patch_2)

        if label == 'mf':
            male_patch = patch_1
        elif label == 'fm':
            male_patch = patch_2
        else:
            raise Exception('Invalid label')

        (cx, cy), angle = pose_predictor.predict(male_patch)

        # predict wing angle
        wing_predictor_input = male_fly_patch(img, mask, (cx, cy), angle)

        if wing_predictor_input is None:
            X.append(data)
            continue

        wing_angle_right, wing_angle_left = wing_predictor.predict(wing_predictor_input)

        if wing_angle_left is None or wing_angle_right is None:
            X.append(data)
            continue

        data[FEATURES.index('right')] = wing_angle_right
        data[FEATURES.index('left')] = wing_angle_left
        X.append(data)

    return X

def main():
    X = load_data()

    joblib.dump(X, X_JOBLIB_NAME)

if __name__ == '__main__':
    main()