import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils

from math import degrees

from cs229.image import img_to_mask, bound_point
from cs229.contour import find_contours, contour_roi, largest_contour
from cs229.annotation import get_annotations

def mask_from_contour(img, contour, color='white'):
    """
    Returns a mask (binary image) selecting only the interior of the contour.
    """

    if color == 'white':
        start_mask = np.zeros(img.shape, dtype=np.uint8)
        contour_color = 255
    elif color == 'black':
        start_mask = np.uint8(255)*np.ones(img.shape, dtype=np.uint8)
        contour_color = 0
    else:
        raise Exception('Invalid color')

    cv2.drawContours(start_mask, [contour], 0, (contour_color, contour_color, contour_color), -1)

    return start_mask

def crop_to_contour(img, contour):
    """
    Returns an ImagePatch containing only the given contour.
    """

    # create a mask of the contour
    mask = mask_from_contour(img, contour)

    # crop to ROI
    roi = contour_roi(contour)
    img = img[roi]
    mask = mask[roi]

    # apply mask
    img = cv2.bitwise_and(img, img, mask=mask)

    return ImagePatch(img=img, mask=mask, ulx=roi[1].start, uly=roi[0].start)

def moments_to_angle(moments):
    """
    Converts OpenCV moments to an angle estimate.  the estimate does have an front/back ambiguity.
    """

    # ref: https://en.wikipedia.org/wiki/Image_moment
    # ref: http://raphael.candelier.fr/?blog=Image%20Moments
    angle = 0.5 * np.arctan(2 * moments['mu11'] / (moments['mu20'] - moments['mu02']))

    if moments['mu20'] < moments['mu02']:
        angle += np.pi / 2

    # negate angle so that increasing angle corresponds to counter-clockwise rotation
    angle = -angle

    return angle

def moments_to_center(moments):
    """
    Returns an estimate of the center-of-mass from the image moments.  Note that this is a floating-point value.
    """

    x_bar = moments['m10']/moments['m00']
    y_bar = moments['m01']/moments['m00']

    return x_bar, y_bar

class ImagePatch:
    def __init__(self, img, mask=None, ulx=0, uly=0):
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

        if self.mask is not None:
            mask = rotate_func(self.mask, degrees(angle))
        else:
            mask = None

        return ImagePatch(img, mask)

    def rotate180(self):
        return self.rotate(np.pi)

    def estimate_angle(self):
        return moments_to_angle(self.moments)

    def estimate_center(self, absolute):
        x, y = moments_to_center(self.moments)

        if absolute:
            x, y = x+self.ulx, y+self.uly

        return (x, y)

    def estimate_axes(self):
        # ref: https://en.wikipedia.org/wiki/Image_moment
        # ref: http://raphael.candelier.fr/?blog=Image%20Moments
        mu_prime_20 = self.moments['mu20']/self.moments['m00']
        mu_prime_02 = self.moments['mu02']/self.moments['m00']
        mu_prime_11 = self.moments['mu11']/self.moments['m00']

        # compute major and minor axes
        MA = np.sqrt(6*(mu_prime_20+mu_prime_02+np.sqrt(4*(mu_prime_11**2)+(mu_prime_20-mu_prime_02)**2)))
        ma = np.sqrt(6*(mu_prime_20+mu_prime_02-np.sqrt(4*(mu_prime_11**2)+(mu_prime_20-mu_prime_02)**2)))

        return MA, ma

    def estimate_aspect_ratio(self):
        MA, ma = self.estimate_axes()

        return MA/ma

    def orient(self, dir='vertical', rotate_angle=None):
        if rotate_angle is None:
            rotate_angle = self.estimate_angle()

        if dir == 'vertical':
            rotate_angle -= np.pi/2
        elif dir == 'horizontal':
            pass
        else:
            raise Exception('Invalid orientation.')

        return self.rotate(rotate_angle)

    def recenter(self, new_width, new_height, old_center=None):
        # define center of original patch
        if old_center is None:
            old_center = bound_point(self.estimate_center(absolute=False), self.img)

        # extract x and y positions of the old center
        old_x = old_center[0]
        old_y = old_center[1]

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

        # compute new image
        new_img[new_window] = self.img[old_window]

        # compute new mask
        if self.mask is not None:
            new_mask[new_window] = self.mask[old_window]
        else:
            new_mask = None

        # compute new upper left-hand corner
        new_ulx = self.ulx + old_x - new_width//2
        new_uly = self.uly + old_y - new_height//2

        return ImagePatch(img=new_img, mask=new_mask, ulx=new_ulx, uly=new_uly)

    def downsample(self, amount):
        # downsample image
        img = self.img[::amount, ::amount]

        # downsample mask
        if self.mask is not None:
            mask = self.mask[::amount, ::amount]
        else:
            mask = None

        return ImagePatch(img=img, mask=mask)

    def flip(self, dir='horizontal'):
        if dir == 'horizontal':
            flip_code = 1
        elif dir == 'vertical':
            flip_code = 0
        else:
            raise Exception('Invalid flip direction.')

        img = cv2.flip(self.img, flip_code)

        if self.mask is not None:
            mask = cv2.flip(self.mask, flip_code)

        return ImagePatch(img=img, mask=mask)

def main():
    # get first annotation
    anno = next(get_annotations())
    image_path = anno.image_path

    # read the image
    img = cv2.imread(image_path, 0)

    # extract contours
    mask = img_to_mask(img)
    contours = find_contours(img, mask=mask, type='core')

    # pick out the largest contour
    contour = largest_contour(contours)

    # crop and rotate
    patch = crop_to_contour(img, contour)
    patch = patch.rotate(patch.estimate_angle()+np.pi/2)

    # display result
    plt.imshow(patch.img)
    plt.show()

if __name__ == "__main__":
    main()