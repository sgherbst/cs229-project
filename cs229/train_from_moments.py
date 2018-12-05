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
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

def random_rows(A, n):
    # ref: https://stackoverflow.com/questions/14262654/numpy-get-random-set-of-rows-from-2d-array
    return A[np.random.choice(A.shape[0], n, replace=False), :]

# def my_kernel(X, Y):
#     X = np.unpackbits(X.astype(np.uint8), axis=1)
#     Y = np.unpackbits(Y.astype(np.uint8), axis=1)
#
#     n = X.shape[1]
#     retval = 1 - X.dot(Y.T)/n - (1-X).dot(1-Y.T)/n
#
#     return retval

def hamming(x, y):
    n = x.shape[0]
    return 1 - x.dot(y)/n - (1-x).dot(1-y)/n

def main():
    print('Loading data...')
    X, y = load_data(os.path.join(top_dir(), 'images', '12-01_10-41-12'), 'legs')

    print(np.std(X, axis=0))

    # clf = SVC(kernel=my_kernel)
    #clf = tree.DecisionTreeClassifier(class_weight='balanced')
    #clf = RandomForestClassifier(class_weight='balanced')

    print('Resampling data...')

    # get all of the positive examples
    X_pos = X[y==1, :]
    n_pos = X_pos.shape[0]

    # randomly choose an equal number of negative examples
    X_neg = X[y==0, :]
    n_neg = X_neg.shape[0]

    n_test = 1000
    pos_hamming = 0
    neg_hamming = 0
    for _ in range(n_test):
        i = randrange(n_pos)
        j = randrange(n_pos)
        pos_hamming += hamming(X_pos[i, :], X_pos[j, :])

        i = randrange(n_pos)
        j = randrange(n_neg)
        neg_hamming += hamming(X_pos[i, :], X_neg[j, :])

    pos_hamming /= n_test
    neg_hamming /= n_test

    print(pos_hamming, neg_hamming)

def main2():
    print('Loading data...')
    X, y = load_data(os.path.join(top_dir(), 'images', '12-01_10-41-12'), 'fabs')

    # clf = SVC(kernel=my_kernel)
    #clf = tree.DecisionTreeClassifier(class_weight='balanced')
    #clf = RandomForestClassifier(class_weight='balanced')

    print('Resampling data...')

    # get all of the positive examples
    X_pos = X[y==1, :]
    n_pos = X_pos.shape[0]
    y_pos = np.ones((n_pos,), dtype=int)

    # randomly choose an equal number of negative examples
    X_neg = X[y==0, :]
    n_neg = X_neg.shape[0]
    y_neg = np.zeros((n_neg,), dtype=int)

    # make new matrices X and y
    X_new = np.concatenate((X_pos, X_neg))
    y_new = np.concatenate((y_pos, y_neg))

    # select training and test split
    print('Making train/test split...')
    X_train, X_test, y_train, y_test = train_test_split(X_new, y_new)

    print('Training...')
    clf.fit(X_train, y_train)

    print('Testing...')
    y_pred = clf.predict(X_test)
    print(confusion_matrix(y_test, y_pred))


if __name__ == '__main__':
    main()