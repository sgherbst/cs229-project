import os
import os.path
import cv2
import numpy as np
import matplotlib.pyplot as plt

from itertools import chain
from math import degrees

from glob import glob

from cs229.files import top_dir
from cs229.annotation import Annotation
from cs229.contour import find_contours
from cs229.image import img_to_mask
from cs229.patch import crop_to_contour
from cs229.contour import contour_label
import joblib

CATEGORIES = ['normal', 'flipped']

def angle_diff(a, b):
    # https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
    return ((a-b) + np.pi) % (2*np.pi) - np.pi

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

def augment_data(hog_patch, label, pos_noise=3, angle_noise=0.05, pixel_noise=3):
    # flip image with 50% probability
    if np.random.rand() < 0.5:
        hog_patch = hog_patch.rotate180()
        label = CATEGORIES[1-CATEGORIES.index(label)]

    # add some angular noise
    hog_patch = hog_patch.rotate(np.random.uniform(-angle_noise, +angle_noise), bound=False)

    # add some translational noise
    hog_patch = hog_patch.translate(np.random.uniform(-pos_noise, +pos_noise),
                                    np.random.uniform(-pos_noise, +pos_noise))

    # add some pixel noise
    hog_patch = hog_patch.add_noise(pixel_noise)

    return hog_patch, label

def load_data(tol_radians=0.1, augment_number=10):
    folders = ['12-04_17-54-43', '12-05-12-43-00']
    folders = [os.path.join(top_dir(), 'images', folder) for folder in folders]
    folders = [glob(os.path.join(folder, '*.json')) for folder in folders]

    files = chain(*folders)

    X = {'male': [], 'female': []}
    y = {'male': [], 'female': []}

    hog = make_hog()

    for f in files:
        anno = Annotation(f)
        img = cv2.imread(anno.image_path, 0)

        mask = img_to_mask(img)
        contours = find_contours(img, mask=mask)

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

            # augment data with reflections, rotations, noise, translation
            for _ in range(augment_number):
                hog_patch_aug, label_aug = augment_data(hog_patch, label)
                X[type].append(patch_to_features(hog_patch_aug, hog))
                y[type].append(CATEGORIES.index(label_aug))

    # assemble features
    X = {k: np.array(v, dtype=float) for k, v in X.items()}

    # assemble labels
    y = {k: np.array(v, dtype=int) for k, v in y.items()}

    return X, y

def main():
    X, y = load_data()

    joblib.dump(X, 'X_orient.joblib')
    joblib.dump(y, 'y_orient.joblib')

if __name__ == '__main__':
    main()