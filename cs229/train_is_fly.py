import os
import os.path
import cv2
import numpy as np

from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from glob import glob

from cs229.files import top_dir
from cs229.annotation import Annotation
from cs229.full_img import FullImage
from cs229.image import img_to_mask

def in_contour(pt, contour):
    return cv2.pointPolygonTest(contour, tuple(pt), False) > 0

def load_data():
    folder = os.path.join(top_dir(), 'images', '12-04_17-54-43')

    neg_examples = []
    pos_examples = []

    for f in glob(os.path.join(folder, '*.json')):
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
                    # discard rare case when the contour includes both
                    anno.warn('Contour contains both flies.')
                else:
                    pos_examples.append(contour)
            else:
                if contains_f:
                    pos_examples.append(contour)
                else:
                    neg_examples.append(contour)

    # assemble features
    X = [cv2.contourArea(contour) for contour in neg_examples+pos_examples]
    X = np.array(X).astype(float)
    X = X.reshape(-1, 1)

    # assemble labels
    y = []
    y += [0 for _ in neg_examples]
    y += [1 for _ in pos_examples]
    y = np.array(y).astype(int)

    return X, y

def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=['not fly', 'is fly']))

if __name__ == '__main__':
    main()