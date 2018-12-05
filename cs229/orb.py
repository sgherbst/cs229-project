import cv2
import cv2.xfeatures2d
import numpy as np
from collections import Iterable

def recenter(v, side, size):
    if v-side < 0:
        v = side
    elif v+side >= size:
        v = size-side-1

    assert (0 <= v-side) and (v+side < size)

    return v

def get_patch(img, i, j, patch_size):
    rows, cols = img.shape

    half_size = patch_size//2
    i = recenter(i, half_size, rows)
    j = recenter(j, half_size, cols)

    return img[i-half_size:i+half_size, j-half_size:j+half_size]

def my_compute(img, kp, patch_size=3):
    des = np.zeros((len(kp), 7), dtype=float)

    for index, point in enumerate(kp):
        i = int(point.pt[1])
        j = int(point.pt[0])

        patch = 255-get_patch(img, i, j, patch_size=patch_size)
        des[index, :] = cv2.HuMoments(cv2.moments(patch)).flatten()

    return kp, des

class MyOrb:
    def __init__(self):
        # self.orb = cv2.ORB_create(
        #     nfeatures=500,
        #     scaleFactor=1.3,
        #     nlevels=3,
        #     edgeThreshold=6,
        #     WTA_K=2,
        #     patchSize=31
        # )
        #

        #self.orb = cv2.ORB_create(patchSize=127)

        kwargs = {'nfeatures': 300, 'nOctaveLayers': 5, 'contrastThreshold': 0.03, 'edgeThreshold': 9, 'sigma': 1.6}
        self.orb = cv2.xfeatures2d.SIFT_create(**kwargs)

    def detect(self, *args, **kwargs):
        self.kp = self.orb.detect(*args, **kwargs)
        self.kp_arr = np.array([[kp.pt[0], kp.pt[1]] for kp in self.kp], dtype='float')

    def compute(self, *args, **kwargs):
        _, des = self.orb.compute(*args, **kwargs)
        #_, des = my_compute(*args, **kwargs)

        return des

    def split_kp(self, pt_or_pts, thresh=6):
        # allow single point or list of points
        if not isinstance(pt_or_pts, Iterable):
            pts = [pt_or_pts]
        else:
            pts = pt_or_pts

        # find all points that are close
        pos = []
        neg_indices = set(range(len(self.kp)))
        for pt in pts:
            dists = np.linalg.norm(self.kp_arr - np.array(pt, dtype='float'), axis=1)

            # remove "insiders" from the set of negative indices
            index, = np.nonzero(dists <= thresh)
            neg_indices -= set(index)

        # split by distance
        pos = [cv2.KeyPoint(x, y, 0) for x, y in pts]
        neg = [self.kp[i] for i in neg_indices]

        return pos, neg
