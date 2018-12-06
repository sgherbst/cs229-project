import os
import os.path
import cv2
import numpy as np

from itertools import chain
from math import degrees

from glob import glob

from cs229.files import top_dir
from cs229.annotation import Annotation
from cs229.full_img import FullImage
from cs229.image import img_to_mask
from cs229.patch import ImagePatch
from cs229.train_contour_classifier import contour_label
import joblib

CATEGORIES = ['normal', 'flipped']

def angle_diff(a, b):
    # https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
    return ((a-b) + np.pi) % (2*np.pi) - np.pi

def hog_patch(img, width=128, height=256, downsamp=2):
    # compute moments
    M = cv2.moments(img)

    # compute center of mass
    in_cx = int(M['m10']/M['m00'])
    in_cy = int(M['m01']/M['m00'])

    # compute center of output image
    out_cx = width//2
    out_cy = height//2

    # compute limits
    left = min(width//2, in_cx)
    right = min((width-1)//2, img.shape[1] - in_cx - 1)
    up = min(height//2, in_cy)
    down = min((height-1)//2, img.shape[0] - in_cy - 1)

    # compute output
    out = np.zeros((height, width), dtype=np.uint8)
    out[out_cy-up:out_cy+down+1, out_cx-left:out_cx+right+1] = img[in_cy-up:in_cy+down+1, in_cx-left:in_cx+right+1]

    # downsample
    out = out[::downsamp, ::downsamp]

    return out

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

def img_to_features(img, hog):
    patch = hog_patch(img)
    descriptor = hog.compute(patch).flatten()

    return descriptor

def load_data(tol_radians=0.1):
    folders = ['12-04_17-54-43', '12-05-12-43-00']
    folders = [os.path.join(top_dir(), 'images', folder) for folder in folders]
    folders = [glob(os.path.join(folder, '*.json')) for folder in folders]

    files = chain(*folders)

    X = {'male': [], 'female': []}
    y = {'male': [], 'female': []}

    hog = make_hog()

    for f in files:
        anno = Annotation(f)
        img = cv2.imread(anno.image_path)

        mask = img_to_mask(img)
        full_image = FullImage(img, mask=mask)

        for contour in full_image.contours:
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
            patch = ImagePatch(full_image.img, contour)

            # compute angle from labels
            label_angle = np.arctan2(abdomen[1] - head[1], head[0] - abdomen[0])

            # find out if the image is flipped or not
            diff = abs(angle_diff(patch.orig_angle, label_angle))

            if diff <= tol_radians:
                label = 'normal'
            elif np.pi-tol_radians <= diff <= np.pi+tol_radians:
                label = 'flipped'
            else:
                anno.warn('Could not properly determine whether image is flipped (diff={:0.1f} degrees)'.format(degrees(diff)))
                continue

            # compute the image descriptor
            X[type] += [img_to_features(patch.img, hog), img_to_features(patch.flipped(), hog)]
            y[type] += [CATEGORIES.index(label), 1-CATEGORIES.index(label)]

    # assemble features
    X = {k: np.array(v).astype(float) for k, v in X.items()}

    # assemble labels
    y = {k: np.array(v).astype(int) for k, v in y.items()}

    return X, y

def main():
    X, y = load_data()

    joblib.dump(X, 'X_orient.joblib')
    joblib.dump(y, 'y_orient.joblib')

if __name__ == '__main__':
    main()