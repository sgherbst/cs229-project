import os
import os.path
import cv2
import numpy as np

from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from itertools import chain

from glob import glob

from cs229.files import top_dir
from cs229.annotation import Annotation
from cs229.full_img import FullImage
from cs229.image import img_to_mask

CATEGORIES = ['none', 'male', 'female', 'both']

def in_contour(pt, contour):
    return cv2.pointPolygonTest(contour, tuple(pt), False) > 0

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

        mp = anno.get('mp')[0]
        fp = anno.get('fp')[0]

        for contour in full_image.contours:
            contains_f = in_contour(fp, contour)
            contains_m = in_contour(mp, contour)

            if contains_m:
                if contains_f:
                    X.append(contour)
                    y.append('both')
                else:
                    # add if this is a good image of a male fly
                    if anno.has('ma') and anno.has('mh'):
                        X.append(contour)
                        y.append('male')
                    else:
                        # bad image; skip
                        pass
            else:
                if contains_f:
                    # add if this is a good image of a female fly
                    if anno.has('fa') and anno.has('fh'):
                        X.append(contour)
                        y.append('female')
                    else:
                        # bad image; skip
                        pass
                else:
                    # discard case where there is not fly
                    X.append(contour)
                    y.append('none')

    # assemble features
    X = [cv2.contourArea(contour) for contour in X]
    X = np.array(X).astype(float)
    X = X.reshape(-1, 1)

    # assemble labels
    y = [CATEGORIES.index(label) for label in y]
    y = np.array(y).astype(int)

    return X, y

def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=CATEGORIES))

if __name__ == '__main__':
    main()