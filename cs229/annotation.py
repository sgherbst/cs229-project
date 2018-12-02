import json
from collections import OrderedDict

class Annotation:
    categories = ['legs', 'wings', 'heads', 'mabs', 'fabs']
    cat_to_num = {cat: num for num, cat in enumerate(categories)}

    @staticmethod
    def one_hot(cat):
        return [1 if cat==v else 0 for v in Annotation.categories]

    def __init__(self, f):
        # general initialization
        self.init_labels()
        self.init_categories()

        # parse the JSON file
        data = json.load(open(f, 'r'))
        self.image_path = data['imagePath']

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
        self.labels = {'ml': [], 'fl': [], 'mj1': [], 'mj2': [], 'fj1': [], 'fj2': [], 'mh': [], 'ma': [],
                       'fh': [], 'fa': [], 'mw': [], 'fw': []}

    def init_categories(self):
        # initialize point lists
        self.legs = []
        self.wings = []
        self.heads = []
        self.mabs = []
        self.fabs = []

        # build tables to look up features
        self.cat_to_lis = {'legs': self.legs, 'wings': self.wings, 'heads': self.heads,
                           'mabs': self.mabs, 'fabs': self.fabs}

    def num_to_lis(self, num):
        return self.cat_to_lis[self.categories[num]]

    def add_labeled_point(self, label, point):
        # add point to dictionary mapping label names to x, y coordinates
        self.labels[label].append(point)

        # add point to specific categories
        if label in ['ml', 'fl']:
            self.legs.append(point)

        if label in ['mw', 'fw']:
            self.wings.append(point)

        if label in ['mh', 'fh']:
            self.heads.append(point)

        if label == 'ma':
            self.mabs.append(point)

        if label == 'fa':
            self.fabs.append(point)

    def sanity_check(self):
        # look for potential issues

        if len(self.legs) > 12 or len(self.labels['ml']) > 6 or len(self.labels['fl']) > 6:
            self.warn('Issue with leg count.')

        if len(self.mabs) > 1 or len(self.fabs) > 1:
            self.warn('Issue with abdomen count.')

        if len(self.wings) > 4 or len(self.labels['mw']) > 2 or len(self.labels['fw']) > 2:
            self.warn('Issue with wing count.')

        if len(self.heads) > 2 or len(self.labels['mh']) > 1 or len(self.labels['fh']) > 1:
            self.warn('Issue with head count.')

        if len(self.labels['mj1']) > 1 or len(self.labels['mj2']) > 1 or len(self.labels['fj1']) > 1 or \
            len(self.labels['fj2']) > 1:
            self.warn('Issue with joint count.')

    def warn(self, msg):
        print('{}: {}'.format(self.path, msg))