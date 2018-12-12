DEBUG = False

import cv2
import numpy as np
import matplotlib.pyplot as plt
import joblib

from math import degrees
from tqdm import tqdm

from cs229.files import get_file
from cs229.annotation import get_annotations
from cs229.contour import find_contours, in_contour
from cs229.image import img_to_mask, bound_point
from cs229.patch import crop_to_contour, ImagePatch, mask_from_contour
from cs229.contour import contour_label
from cs229.util import angle_diff, report_labels_regression

X_JOBLIB_NAME = 'X_wing.joblib'
Y_JOBLIB_NAME = 'y_wing.joblib'

def get_rotation_matrix(theta):
    # ref: https://scipython.com/book/chapter-6-numpy/examples/creating-a-rotation-matrix-in-numpy/
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,-s], [s, c]], dtype=float)

def male_fly_patch(img, mask, body_center, rotate_angle, crop_size=400):
    # legalize body_center
    body_center = bound_point((body_center[0], body_center[1]), img)

    # create a patch centered on the fly
    fly_patch = ImagePatch(img, mask)
    fly_patch = fly_patch.recenter(crop_size, crop_size, body_center)

    # extract contours of the patch
    contours = find_contours(fly_patch.img, fly_patch.mask, type='wings')

    # mask out everything except the contour containing the fly
    fly_center = (crop_size//2, crop_size//2)
    for contour in sorted(contours, key=lambda x: len(x)):
        if in_contour(fly_center, contour):
            # create new mask for fly region (still keeping circular ROI)
            fly_mask = mask_from_contour(fly_patch.img, contour)
            fly_patch.mask = cv2.bitwise_and(fly_patch.mask, fly_mask)

            # apply mask to the fly_patch image
            fly_patch.img = cv2.bitwise_and(fly_patch.img, fly_patch.img, mask=fly_patch.mask)
            break
    else:
        # return None if no contour contains the fly center
        return None

    # orient fly vertically
    fly_patch = fly_patch.orient('vertical', rotate_angle=rotate_angle)

    # return the resulting fly patch
    return fly_patch

def make_hog_patch(fly_patch):
    # crop image to a region of interest around the fly
    cx, cy = fly_patch.img.shape[0]//2, fly_patch.img.shape[1]//2
    window = (slice(cy - 68, cy + 124), slice(cx, cx + 160))
    hog_patch = ImagePatch(fly_patch.img[window], fly_patch.mask[window])

    # downsample patch
    hog_patch = hog_patch.downsample(2)

    # return the resulting fly patch
    return hog_patch

def make_hog():
    winSize = (80, 96)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

    return hog

def patch_to_features(hog_patch, hog):
    return hog.compute(hog_patch.img).flatten()

def load_data(tol_radians=0.1):
    hog = make_hog()

    X = []
    y = []

    img_count = 0
    hand_labeled_count = 0

    for anno in tqdm(get_annotations()):
        img = cv2.imread(anno.image_path, 0)

        mask = img_to_mask(img)
        contours = find_contours(img, mask=mask, type='core')

        for contour in contours:
            type = contour_label(anno, contour)
            if type == 'male':
                break
        else:
            continue

        if anno.count('mw') == 2 and anno.has('mh') and anno.has('ma') and anno.has('mp2'):
            mh = anno.get('mh')[0]
            ma = anno.get('ma')[0]
            mp2 = anno.get('mp2')[0]
        else:
            continue

        # make patch, which will compute the angle from the image
        body_patch = crop_to_contour(img, contour)

        # compute angle from labels
        label_angle = np.arctan2(ma[1] - mh[1], mh[0] - ma[0])

        # find out if the image is flipped or not
        rotate_angle = body_patch.estimate_angle()
        diff = abs(angle_diff(rotate_angle, label_angle))

        if diff <= tol_radians:
            pass
        elif np.pi-tol_radians <= diff <= np.pi+tol_radians:
            rotate_angle = rotate_angle + np.pi
        else:
            anno.warn('Could not properly determine whether image is flipped (diff={:0.1f} degrees)'.format(degrees(diff)))
            continue

        # find center of fly
        body_center = body_patch.estimate_center(absolute=True)

        # create patch centered on fly
        fly_patch = male_fly_patch(img, mask, body_center, rotate_angle)
        if fly_patch is None:
            continue

        origin = bound_point((body_center[0], body_center[1]), img)
        mp2_rel = [mp2[0]-origin[0], origin[1]-mp2[1]]

        rot_mat = get_rotation_matrix(np.pi/2-rotate_angle)
        mp2_rot = rot_mat.dot(mp2_rel)

        wings = []
        for mw in anno.get('mw'):
            wing_rel = [mw[0]-origin[0], origin[1]-mw[1]]
            wing_rot = rot_mat.dot(wing_rel) - mp2_rot
            angle = np.arctan(abs(wing_rot[0])/abs(wing_rot[1]))

            wings.append({'x': wing_rot[0], 'angle': angle})
        if len(wings) != 2:
            anno.warn('Length of wings is not 2 for some reason, skipping...')
            continue

        if wings[0]['x'] > wings[1]['x']:
            # wing 0 is right, wing 1 is left
            right_wing_angle = wings[0]['angle']
            left_wing_angle = wings[1]['angle']
        else:
            # wing 1 is right, wing 0 is left
            right_wing_angle = wings[1]['angle']
            left_wing_angle = wings[0]['angle']

        # create a hog patches for both wings
        data = []
        data.append({'hog_patch': make_hog_patch(fly_patch), 'angle': right_wing_angle})
        data.append({'hog_patch': make_hog_patch(fly_patch.flip('horizontal')), 'angle': left_wing_angle})

        # add data to X and y
        for datum in data:
            X.append(patch_to_features(datum['hog_patch'], hog))
            y.append(datum['angle'])
            hand_labeled_count += 1

            if DEBUG:
                plt.imshow(datum['hog_patch'].img)
                plt.show()

        # increment img_count to indicate that this file was actually used
        img_count += 1

    # assemble features
    X = np.array(X, dtype=float)

    # assemble labels
    y = np.array(y, dtype=float)

    print('Used {} annotated images.'.format(img_count))
    print('Used {} hand-labeled wings.'.format(hand_labeled_count))

    report_labels_regression(np.degrees(y),
                             filename=get_file('output', 'graphs', 'labels_wing.eps'),
                             units='Wing Angle (degrees)')

    return X, y

def main():
    X, y = load_data()

    joblib.dump(X, get_file('output', 'data', X_JOBLIB_NAME))
    joblib.dump(y, get_file('output', 'data', Y_JOBLIB_NAME))

if __name__ == '__main__':
    main()