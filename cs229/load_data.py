import cv2
import numpy as np
import os.path
import os
import pickle
from glob import glob

from cs229.files import open_image, open_video, top_dir
from cs229.display import open_window, show_image, make_trackbar, get_trackbar
from cs229.image import img_to_mask
from cs229.util import FpsMon, TickTock
from cs229.annotation import Annotation
from cs229.orb import MyOrb

def process(orb, lis, img):
    # get indices of positive and negative samples
    pos_kp, neg_kp = orb.split_kp(lis)

    # compute descriptor for each keypoint
    X = orb.compute(img, pos_kp+neg_kp)

    # assign labels
    y = np.zeros((len(pos_kp)+len(neg_kp),), dtype=int)
    y[:len(pos_kp)] = 1

    return X, y

def load_data(folder, cat):
    X = []
    y = []

    # create ORB detector
    orb = MyOrb()

    for file in glob(os.path.join(folder, '*.json')):
        # load annotations
        anno = Annotation(file)

        # load image
        img = cv2.imread(os.path.join(folder, anno.image_path), 0)
        mask = img_to_mask(img)

        # detect the keypoints
        orb.detect(img, mask=mask)

        # update data
        X_new, y_new = process(orb, anno.cat_to_lis[cat], img)
        X.append(X_new)
        y.append(y_new)

    X = np.concatenate(X)
    X = np.unpackbits(X.astype(np.uint8), axis=1)

    y = np.concatenate(y)

    return X, y

def main():
    X, y = load_data(os.path.join(top_dir(), 'images', '12-01_10-41-12'), 'legs')
    print(X.shape, y.shape)

if __name__ == '__main__':
    main()