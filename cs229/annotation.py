import json
import numpy as np
import os
import os.path
from glob import glob
from itertools import chain

from cs229.files import get_dir

def project(a, b, c):
    """
    Project (c-a) onto (b-a) and return the projection coefficient.  If this coefficient is between 0 and 1, then
    c can be interpreted as being between a and b (even though it may not lie directly on the line between them)
    """

    # make numpy arrays of the three points
    a = np.array(a).astype(float)
    b = np.array(b).astype(float)
    c = np.array(c).astype(float)

    # unit vector from a to b
    u = (b-a)/np.linalg.norm(b-a)

    # project c onto the vector
    p = (c-a).dot(u)

    # return projection coefficient normalized to length between a and b
    return p/np.linalg.norm(b-a)

class Annotation:
    def __init__(self, f):
        # save path to JSON file
        self.json_path = f

        # initialize labels
        self.init_labels()

        # parse the JSON file
        data = json.load(open(f, 'r'))
        self.image_path = os.path.join(os.path.dirname(self.json_path), data['imagePath'])

        # loop through all points
        for shape in data['shapes']:
            # get the label name and point location
            label = shape['label']
            point = shape['points'][0]

            # add the point
            self.add_labeled_point(label, point)

        # run sanity check to identify possible errors
        self.sanity_check()

    def init_labels(self):
        self.labels = {}

    def count(self, label):
        return len(self.get(label)) if self.has(label) else 0

    def get(self, label):
        return self.labels[label]

    def has(self, label):
        return label in self.labels

    def check_features(self, **kwargs):
        for key, val in kwargs.items():
            if not len(self.labels[key]) == val:
                return False
        return True

    def add_labeled_point(self, label, point):
        if not self.has(label):
            self.labels[label] = []

        self.labels[label].append(point)

    def sanity_check(self):
        # look for potential issues

        if self.count('fp') != 1 or self.count('mp') != 1:
            self.warn('Must have one male and one female fly.')
            return

        if not ((self.count('fh') == 0 and self.count('fa') == 0) or
                (self.count('fh') == 1 and self.count('fa') == 1)):
            self.warn('Must define both female head and abdomen or neither.')
            return

        if not ((self.count('mh') == 0 and self.count('ma') == 0) or
                (self.count('mh') == 1 and self.count('ma') == 1)):
            self.warn('Must define both male head and abdomen or neither.')
            return

        if self.count('fh') == 1 and self.count('fa') == 1:
            fa = self.get('fa')[0]
            fh = self.get('fh')[0]
            fp = self.get('fp')[0]

            if not (0 <= project(fa, fh, fp) <= 1):
                self.warn('fp not between fa and fh')
                return

        if self.count('mh') == 1 and self.count('ma') == 1:
            ma = self.get('ma')[0]
            mh = self.get('mh')[0]
            mp = self.get('mp')[0]

            if not (0 <= project(ma, mh, mp) <= 1):
                self.warn('mp not between ma and mh')
                return

        if self.count('mp2') == 0:
            pass
        elif self.count('mp2') == 1:
            if not (self.count('mp') == 1 and self.count('ma') == 1 and self.count('mh') == 1):
                self.warn('Must define mp, ma, and mh to use mp2.')
                return

            ma = self.get('ma')[0]
            mh = self.get('mh')[0]
            mp = self.get('mp')[0]
            mp2 = self.get('mp2')[0]

            if not (0 <= project(mh, ma, mp2) <= project(mh, ma, mp)):
                self.warn('mp2 must be between mh and mp')
                return
        else:
            self.warn('Cannot define multiple mp2 points.')
            return

        if self.count('mw') == 0:
            pass
        elif self.count('mw') == 2:
            if not (self.count('mp') == 1 and self.count('ma') == 1 and self.count('mh') == 1 and
                    self.count('mp2')==1):
                self.warn('Must define mp, mp2, ma, and mh to define wings.')
                return
        else:
            self.warn('Must define zero or two wings.')
            return

    def warn(self, msg):
        print('{}: {}'.format(self.image_path, msg))


def get_annotations():
    """
    Returns an iterator over all annotation files in the subdirectories of input/images
    """

    # get subfolders of input/images directory
    # ref: https://stackoverflow.com/questions/973473/getting-a-list-of-all-subdirectories-in-the-current-directory
    subfolders = [f.path for f in os.scandir(get_dir('input', 'images')) if f.is_dir()]

    # get all JSON files from subfolders
    files = [glob(os.path.join(subfolder, '*.json')) for subfolder in subfolders]

    return (Annotation(f) for f in chain(*files))


def main():
    for anno in get_annotations():
        pass

if __name__ == '__main__':
    main()