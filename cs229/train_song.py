import numpy as np
import matplotlib.pyplot as plt

from cs229.load_data_song import X_JOBLIB_NAME, FEATURES
from scipy import signal

import joblib

WIN_SIZE = 16
CLF_JOBLIB_NAME = 'clf_song.joblib'

HAMMING_WINDOW = np.hamming(WIN_SIZE)

def freq_repr(x):
    fft = np.fft.fft(x*HAMMING_WINDOW)
    fft = fft[0:len(x)//2 + 1]
    fft = np.abs(fft)

    return fft

def filter(x, alpha=0.95):
    x = np.array(x)
    z, _ = signal.lfilter([1 - alpha], [1, -alpha], x, zi=[x[0]])

    return z

def train(X):
    # chop into snippets without None
    right_values = [x[FEATURES.index('right')] for x in X]
    left_values = [x[FEATURES.index('left')] for x in X]

    start = 0
    r_snippet = []
    l_snippet = []
    for k, (r, l) in enumerate(zip(right_values, left_values)):
        if r is None or l is None:
            if r_snippet and l_snippet:
                x_axis = list(range(start, k))
                plt.plot(x_axis, np.degrees(r_snippet), '-b')
                plt.plot(x_axis, -np.degrees(l_snippet), '-r')
            start = k+1
            r_snippet = []
            l_snippet = []
        else:
            r_snippet.append(r)
            l_snippet.append(l)

    plt.legend(['Right Wing', 'Left Wing'])
    plt.xlabel('Frame #')
    plt.ylabel('Wing Angle (degrees)')

    plt.savefig('wave_song.eps'.format(type), bbox_inches='tight')
    plt.clf()

def main():
    X = joblib.load(X_JOBLIB_NAME)

    train(X)

if __name__ == '__main__':
    main()