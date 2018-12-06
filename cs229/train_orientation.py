import os
import os.path
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from itertools import chain
from math import degrees

from glob import glob

from cs229.files import top_dir
from cs229.annotation import Annotation
from cs229.full_img import FullImage
from cs229.image import img_to_mask
from cs229.patch import ImagePatch
from cs229.train_contour_classifier import contour_label
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

from cs229.load_data_orientation import CATEGORIES

def main():
    X = np.load('X_orient.npy')
    y= np.load('y_orient.npy')

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = make_pipeline(StandardScaler(), PCA(n_components=2), LogisticRegression())

    clf = clf.fit(X_train, y_train)

    scaler = clf.named_steps['standardscaler']
    pca_std = clf.named_steps['pca']
    X_train_std_transformed = pca_std.transform(scaler.transform(X_train))
    for l, c, m in zip(range(2), ('blue', 'red'), ('o', 'x')):
        plt.scatter(X_train_std_transformed[y_train == l, 0],
                    X_train_std_transformed[y_train == l, 1],
                    color=c,
                    label='class %s' % l,
                    alpha=0.5,
                    marker=m
                    )

    plt.legend(loc='upper right')
    plt.grid()
    plt.show()

    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=CATEGORIES))

if __name__ == '__main__':
    main()