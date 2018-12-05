import cv2
import os
import os.path
import matplotlib.pyplot as plt
import numpy as np
import imutils

from cs229.files import top_dir
from cs229.image import img_to_mask
from cs229.full_img import FullImage
from math import degrees

def crop_to_contour(img, contour):
    # create a mask of the contour
    mask = np.zeros(img.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], 0, (255, 255, 255), -1)

    # get bounding box coordinates
    xmin = np.min(contour[:, 0, 1])
    xmax = np.max(contour[:, 0, 1])
    ymin = np.min(contour[:, 0, 0])
    ymax = np.max(contour[:, 0, 0])

    # mask and crop
    out = cv2.bitwise_and(img[xmin:xmax + 1, ymin:ymax + 1],
                          img[xmin:xmax + 1, ymin:ymax + 1],
                          mask=mask[xmin:xmax + 1, ymin:ymax + 1])

    return out

def moments_to_angle(moments):
    angle = 0.5 * np.arctan(2 * moments['mu11'] / (moments['mu20'] - moments['mu02']))

    if moments['mu20'] < moments['mu02']:
        angle += np.pi / 2

    angle = -angle

    return angle

def moments_to_center(moments):
    x_bar = moments['m10']/moments['m00']
    y_bar = moments['m01']/moments['m00']

    return x_bar, y_bar

class ImagePatch:
    def __init__(self, img, contour):
        patch = crop_to_contour(img, contour)

        moments = cv2.moments(patch)
        self.orig_angle = moments_to_angle(moments)

        self.img = imutils.rotate_bound(patch, degrees(self.orig_angle))

    def flipped(self):
        return imutils.rotate_bound(self.img, 180)

def main():
    # read the image
    image_path = os.path.join(top_dir(), 'images', '12-01_10-41-12', '12.bmp')
    img = cv2.imread(image_path)

    # extract contours
    mask = img_to_mask(img)
    full_img = FullImage(img, mask=mask)

    # pick out the largest contour
    contour = max(full_img.contours, key=lambda x: cv2.contourArea(x))

    # crop and rotate
    patch = ImagePatch(full_img.img, contour)

    # display result
    plt.imshow(patch.img)
    plt.show()

if __name__ == "__main__":
    main()