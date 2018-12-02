import cv2
import numpy as np
from collections import Iterable

class MyOrb:
    def __init__(self):
        self.orb = cv2.ORB_create(
            nfeatures=500,
            scaleFactor=1.3,
            nlevels=3,
            edgeThreshold=6,
            WTA_K=2,
            patchSize=31
        )

    def detect(self, *args, **kwargs):
        self.kp = self.orb.detect(*args, **kwargs)
        self.kp_arr = np.array([[kp.pt[0], kp.pt[1]] for kp in self.kp], dtype='float')

    def compute(self, *args, **kwargs):
        _, des = self.orb.compute(*args, **kwargs)
        return des

    def closest_kp(self, pt_or_pts, max_dist=5):
        if not isinstance(pt_or_pts, Iterable):
            pts = [pt_or_pts]
        else:
            pts = pt_or_pts

        retval = []
        for pt in pts:
            dists = np.linalg.norm(self.kp_arr - np.array(pt, dtype='float'), axis=1)
            index = np.argmin(dists)
            if dists[index] <= max_dist:
                retval.append(self.kp[index])

        if not isinstance(pt_or_pts, Iterable):
            return retval[0] if retval else None
        else:
            return retval

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

            # define the positive example as the closest keypoint to the labeled point
            # only add the keypoint if it would be considered an "insider"
            index = np.argmin(dists)
            if dists[index] <= thresh:
                pos.append(self.kp[index])

            # remove "insiders" from the set of negative indices
            index, = np.nonzero(dists <= thresh)
            neg_indices -= set(index)

        # split by distance
        return pos, [self.kp[i] for i in neg_indices]
