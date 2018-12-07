import os
import os.path
import cv2
import numpy as np

from itertools import chain

from glob import glob

from cs229.files import top_dir
from cs229.annotation import Annotation
from cs229.full_img import FullImage
from cs229.image import img_to_mask

import joblib

CATEGORIES = ['neither', 'male', 'female', 'both']

def in_contour(pt, contour):
    return cv2.pointPolygonTest(contour, tuple(pt), False) > 0

def contour_to_features(contour):
    return np.array([cv2.contourArea(contour)], dtype=float)

def contour_label(anno, contour):
    contains_m = in_contour(anno.get('mp')[0], contour)
    contains_f = in_contour(anno.get('fp')[0], contour)

    if contains_m:
        if contains_f:
            return 'both'
        else:
            if anno.has('ma') and anno.has('mh'):
                return 'male'
            else:
                # bad image; skip
                return None
    else:
        if contains_f:
            if anno.has('fa') and anno.has('fh'):
                return 'female'
            else:
                # bad image; skip
                pass
        else:
            return 'neither'

def load_data():
    folders = ['12-04_17-54-43', '12-05-12-43-00']
    folders = [os.path.join(top_dir(), 'images', folder) for folder in folders]
    folders = [glob(os.path.join(folder, '*.json')) for folder in folders]

    files = chain(*folders)

    X = []
    y = []

    for f in files:
        anno = Annotation(f)
        img = cv2.imread(anno.image_path)

        mask = img_to_mask(img)
        full_image = FullImage(img, mask=mask)

        for contour in full_image.contours:
            label = contour_label(anno, contour)
            if label is not None:
                X.append(contour_to_features(contour))
                y.append(CATEGORIES.index(label))

    # assemble features
    X = np.array(X, dtype=float)

    # assemble labels
    y = np.array(y, dtype=int)

    return X, y

def main():
    X, y = load_data()

    joblib.dump(X, 'X_contour.joblib')
    joblib.dump(y, 'y_contour.joblib')

if __name__ == '__main__':
    main()