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
    ymin = np.min(contour[:, 0, 1])
    ymax = np.max(contour[:, 0, 1])
    xmin = np.min(contour[:, 0, 0])
    xmax = np.max(contour[:, 0, 0])

    # crop to window
    window = (slice(ymin, ymax+1), slice(xmin, xmax+1))
    img = img[window]
    mask = mask[window]

    # apply mask
    img = cv2.bitwise_and(img, img, mask=mask)

    return ImagePatch(img=img, mask=mask, ulx=xmin, uly=ymin)

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
    def __init__(self, img, mask, ulx=0, uly=0):
        self.img = img
        self.mask = mask
        self.ulx = ulx
        self.uly = uly

        self._moments = None

    # memoized image moments
    @property
    def moments(self):
        if self._moments is None:
            self._moments = cv2.moments(self.img)

        return self._moments

    def rotate(self, angle, bound=True):
        rotate_func = imutils.rotate_bound if bound else imutils.rotate
        img = rotate_func(self.img, degrees(angle))
        mask = rotate_func(self.mask, degrees(angle))

        return ImagePatch(img, mask)

    def translate(self, x, y):
        img = imutils.translate(self.img, x, y)
        mask = imutils.translate(self.mask, x, y)

        return ImagePatch(img, mask)

    def rotate180(self):
        return self.rotate(np.pi)

    def estimate_angle(self):
        return moments_to_angle(self.moments)

    def estimate_center(self):
        return moments_to_center(self.moments)

    def orient(self, dir='vertical'):
        rotate_angle = self.estimate_angle()

        if dir == 'vertical':
            rotate_angle -= np.pi/2
        elif dir == 'horizontal':
            pass
        else:
            raise Exception('Invalid orientation.')

        return self.rotate(rotate_angle)

    def add_noise(self, amount):
        # add noise to the whole image
        img = self.img.astype(float) + np.random.uniform(-amount, +amount, size=self.img.shape)
        img = np.clip(img, 0, 255).astype(np.uint8)

        # apply mask
        img = cv2.bitwise_and(img, img, mask=self.mask)

        return ImagePatch(img=img, mask=self.mask)

    def recenter(self, new_width, new_height):
        # define center of original patch
        old_x, old_y = self.estimate_center()
        old_x = np.clip(np.round(old_x), 0, self.img.shape[1] - 1).astype(int)
        old_y = np.clip(np.round(old_y), 0, self.img.shape[0] - 1).astype(int)

        # define center of new patch
        new_x, new_y = new_width // 2, new_height // 2

        # compute limits
        left = min(old_x, new_x)
        up = min(old_y, new_y)
        right = min(new_width - new_x, self.img.shape[1] - old_x)
        down = min(new_height - new_y, self.img.shape[0] - old_y)

        # compute cropping windows
        new_window = (slice(new_y-up, new_y+down), slice(new_x-left, new_x+right))
        old_window = (slice(old_y-up, old_y+down), slice(old_x-left, old_x+right))

        # initialize new mask and image to zeros
        new_img = np.zeros((new_height, new_width), dtype=np.uint8)
        new_mask = np.zeros((new_height, new_width), dtype=np.uint8)

        # compute new image and mask
        new_img[new_window] = self.img[old_window]
        new_mask[new_window] = self.mask[old_window]

        return ImagePatch(img=new_img, mask=new_mask)

    def downsample(self, amount):
        img = self.img[::amount, ::amount]
        mask = self.mask[::amount, ::amount]

        return ImagePatch(img=img, mask=mask)

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
    patch = crop_to_contour(full_img.img, contour)
    patch = patch.rotate(patch.estimate_angle()+np.pi/2)

    # display result
    plt.imshow(patch.img)
    plt.show()

if __name__ == "__main__":
    main()