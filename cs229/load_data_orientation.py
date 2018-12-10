import cv2
import numpy as np
import matplotlib.pyplot as plt

from math import degrees

from cs229.files import get_annotation_files
from cs229.annotation import Annotation
from cs229.contour import find_core_contours
from cs229.image import img_to_mask
from cs229.patch import crop_to_contour
from cs229.contour import contour_label
from cs229.util import angle_diff, report_labels
import joblib

CATEGORIES = ['normal', 'flipped']

def make_hog_patch(patch):
    patch = patch.orient('vertical')
    patch = patch.recenter(new_width=128, new_height=256)
    patch = patch.downsample(2)

    return patch

def make_hog():
    # winSize = (64, 128)
    # blockSize = (16, 16)
    # blockStride = (8, 8)
    # cellSize = (8, 8)
    # nbins = 9
    #
    # hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

    hog = cv2.HOGDescriptor()

    return hog

def patch_to_features(hog_patch, hog):
    return hog.compute(hog_patch.img).flatten()

def load_data(tol_radians=0.1):
    X = {'male': [], 'female': []}
    y = {'male': [], 'female': []}

    hog = make_hog()

    img_count = 0
    hand_labeled_count = 0

    for f in get_annotation_files():
        # keep track of whether any data is actually used from this file
        used_file = False

        anno = Annotation(f)
        img = cv2.imread(anno.image_path, 0)

        mask = img_to_mask(img)
        contours = find_core_contours(img, mask=mask)

        for contour in contours:
            type = contour_label(anno, contour)

            if type == 'male' and anno.has('ma') and anno.has('mh'):
                type = 'male'
                head = anno.get('mh')[0]
                abdomen = anno.get('ma')[0]
            elif type == 'female' and anno.has('fa') and anno.has('fh'):
                type = 'female'
                head = anno.get('fh')[0]
                abdomen = anno.get('fa')[0]
            else:
                continue

            # make patch, which will compute the angle from the image
            patch = crop_to_contour(img, contour)

            # compute angle from labels
            label_angle = np.arctan2(abdomen[1] - head[1], head[0] - abdomen[0])

            # find out if the image is flipped or not
            diff = abs(angle_diff(patch.estimate_angle(), label_angle))

            if diff <= tol_radians:
                label = 'normal'
            elif np.pi-tol_radians <= diff <= np.pi+tol_radians:
                label = 'flipped'
            else:
                anno.warn('Could not properly determine whether image is flipped (diff={:0.1f} degrees)'.format(degrees(diff)))
                continue

            # orient patch vertically
            hog_patch = make_hog_patch(patch)

            # add original data
            X[type].append(patch_to_features(hog_patch, hog))
            y[type].append(CATEGORIES.index(label))

            hand_labeled_count += 1
            used_file = True

            # augment data by flipping the image and inverting the label
            hog_patch_flipped = hog_patch.rotate180()
            label_flipped = CATEGORIES[1 - CATEGORIES.index(label)]

            # add this additional feature
            X[type].append(patch_to_features(hog_patch_flipped, hog))
            y[type].append(CATEGORIES.index(label_flipped))

        if used_file:
            img_count += 1

    # assemble features
    X = {k: np.array(v, dtype=float) for k, v in X.items()}

    # assemble labels
    y = {k: np.array(v, dtype=int) for k, v in y.items()}

    print('Used {} annotated images.'.format(img_count))
    print('Used {} hand-labeled flies.'.format(hand_labeled_count))
    print()

    print('Male classifier:')
    report_labels(CATEGORIES, y['male'])
    print()

    print('Female classifier:')
    report_labels(CATEGORIES, y['female'])

    return X, y

def main():
    X, y = load_data()

    joblib.dump(X, 'X_orient.joblib')
    joblib.dump(y, 'y_orient.joblib')

if __name__ == '__main__':
    main()