import cv2
import numpy as np
import os.path
import os
import pickle
from glob import glob
from random import randrange

from cs229.files import open_image, open_video, top_dir
from cs229.display import open_window, show_image, make_trackbar, get_trackbar
from cs229.image import img_to_mask
from cs229.util import FpsMon, TickTock
from cs229.annotation import Annotation
from cs229.orb import MyOrb
from cs229.load_data import load_data
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def main():
    print('Loading data...')
    X, y = load_data(os.path.join(top_dir(), 'images', '12-01_10-41-12'))

    # prepare features
    X = [cv2.HuMoments(v).flatten() for v in X]
    X = np.array(X).astype(float)
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)

    # prepare labels
    y = np.array(y).astype(int)

    # split into training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # make logistic regression model
    clf = LogisticRegression()

    # train
    print('Training...')
    clf.fit(X_train, y_train)

    # test
    print('Testing...')
    y_pred = clf.predict(X_test)
    print(confusion_matrix(y_test, y_pred))

if __name__ == '__main__':
    main()