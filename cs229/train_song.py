import numpy as np
import matplotlib.pyplot as plt

from cs229.load_data_song import X_JOBLIB_NAME, FEATURES
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import signal
from sklearn.pipeline import make_pipeline

import joblib

WIN_SIZE = 64
CLF_JOBLIB_NAME = 'clf_song.joblib'

HAMMING_WINDOW = np.hamming(WIN_SIZE)

def freq_repr(x):
    fft = np.fft.fft(x*HAMMING_WINDOW)
    fft = fft[0:len(x)//2 + 1]
    fft = np.abs(fft)

    return fft

def prepare(X):
    # chop into snippets without None
    right_values = [x[FEATURES.index('right')] for x in X]
    left_values = [x[FEATURES.index('left')] for x in X]

    data = []
    for k in range(len(X)-WIN_SIZE+1):
        r_seg = right_values[k:k+WIN_SIZE]
        l_seg = left_values[k:k+WIN_SIZE]

        if None in r_seg or None in l_seg:
            continue

        r_seg = np.array(r_seg)
        l_seg = np.array(l_seg)
        a_seg = np.array(r_seg - l_seg)

        #datum = np.hstack((freq_repr(r_seg)))
        datum = np.hstack((freq_repr(r_seg), freq_repr(l_seg), freq_repr(a_seg)))
        data.append(datum)

    X_t = np.array(data, dtype=float)

    return X_t

def train(X):
    print('Preparing data...')
    X = prepare(X)

    # apply PCA to data
    print('Applying PCA...')

    clf = make_pipeline(StandardScaler(), PCA(n_components=50))
    clf.fit(X)

    plt.plot(np.cumsum(clf.named_steps['pca'].explained_variance_ratio_))
    plt.xlabel('Number of PCA components')
    plt.ylabel('Total explained variance ratio')
    plt.grid()
    plt.show()

    # plot principal component vs angle
    X_new = clf.transform(X)

    plt.scatter(X_new[:, 0].flatten(), X_new[:, 1].flatten(), s=0.25)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid()

    plt.show()

def main():
    X = joblib.load(X_JOBLIB_NAME)

    train(X)

if __name__ == '__main__':
    main()