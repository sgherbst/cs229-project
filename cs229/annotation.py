import json
import numpy as np
from glob import glob
import os
import os.path

from cs229.files import top_dir

def project(a, b, c):
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

def main():
    folders = []
    folders.append(os.path.join(top_dir(), 'images', '12-04_17-54-43'))
    folders.append(os.path.join(top_dir(), 'images', '12-05-12-43-00'))
    folders.append(os.path.join(top_dir(), 'images', '12-07_16_45_00'))
    folders.append(os.path.join(top_dir(), 'images', '12-08_11-15-00'))
    folders.append(os.path.join(top_dir(), 'images', '12-08_22_00_00'))

    for folder in folders:
        for f in glob(os.path.join(folder, '*.json')):
            Annotation(f).image_path

if __name__ == '__main__':
    main()