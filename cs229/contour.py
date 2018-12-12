import cv2
import matplotlib.pyplot as plt
import numpy as np

from cs229.image import img_to_mask
from cs229.annotation import get_annotations

# to avoid computing the same erosion kernel over and over again...
ERODE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))

def in_contour(pt, contour):
    """
    Return True if pt=(x,y) is contained inside the given contour.
    """

    return cv2.pointPolygonTest(contour, tuple(pt), False) > 0

def contour_roi(contour):
    xmin = np.min(contour[:, 0, 0])
    xmax = np.max(contour[:, 0, 0])
    ymin = np.min(contour[:, 0, 1])
    ymax = np.max(contour[:, 0, 1])

    return (slice(ymin, ymax+1), slice(xmin, xmax+1))

def largest_contour(contours):
    return max(contours, key=lambda contour: cv2.contourArea(contour))

def contour_label(anno, contour):
    """
    Given a contour and image annotation, returns when the contour contains 0, 1, or 2 flies.
    """

    contains_m = in_contour(anno.get('mp')[0], contour)
    contains_f = in_contour(anno.get('fp')[0], contour)

    if contains_m:
        if contains_f:
            return 'both'
        else:
            return 'male'
    else:
        if contains_f:
            return 'female'
        else:
            return 'neither'

def find_contours(img, mask=None, fly_color='black', type='core'):
    """
    Returns contours after applying an image thresholding scheme.
    """

    # threshold image
    if type == 'core':
        bw = threshold_core(img=img, mask=mask, fly_color=fly_color)
    elif type == 'wings':
        bw = threshold_wings(img=img, mask=mask, fly_color=fly_color)
    else:
        raise Exception('Invalid type.')

    # extract contours
    _, contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return contours
    return contours

def threshold_wings(img, mask=None, fly_color='black', mean_offset=5):
    """
    Blur, threshold, then erode the given image in an attempt to produce a binary image in which flies are separated
    from each other and the walls/background.  Unlike threshold_core, care must be taken not to remove the wings.
    """

    # determine thresholding type
    if fly_color == 'black':
        thresh_type = cv2.THRESH_BINARY_INV
    elif fly_color == 'white':
        thresh_type = cv2.THRESH_BINARY
    else:
        raise Exception('Invalid fly color.')

    # apply median filter
    blurred = cv2.medianBlur(img, 5)
    mean_in_roi = np.sum((mask==255)*img)/np.sum(mask==255)

    # threshold image a bit below the median
    _, bw = cv2.threshold(blurred, mean_in_roi-mean_offset, 255, thresh_type)

    # erode image
    eroded = cv2.erode(bw, ERODE_KERNEL, iterations=1)

    # mask to ROI
    masked = cv2.bitwise_and(eroded, mask)

    return masked

def threshold_core(img, mask=None, fly_color='black', thresh_int=115):
    # Apply a fixed threshold to the given image to search for flies.  The threshold is set to a low value
    # so that only the "core" of the fly (i.e., head and abdomen) is visible.

    # determine thresholding type
    if fly_color == 'black':
        thresh_type = cv2.THRESH_BINARY_INV
    elif fly_color == 'white':
        thresh_type = cv2.THRESH_BINARY
    else:
        raise Exception('Invalid fly color.')

    # create the black and white image
    _, bw = cv2.threshold(img, thresh_int, 255, thresh_type)

    # apply mask to thresholded image
    masked = cv2.bitwise_and(bw, mask)

    # return the thresholded and masked image
    return masked

def main():
    # get the very first annotation file and use the corresponding
    anno = next(get_annotations())
    image_path = anno.image_path

    # read the image
    img = cv2.imread(image_path, 0)
    mask = img_to_mask(img)

    # display
    plt.imshow(img)
    plt.show()

    # find fly
    contour = largest_contour(find_contours(img, mask))
    roi = contour_roi(contour)

    # display
    bw = threshold_core(img[roi], mask[roi])
    plt.imshow(bw)
    plt.show()

if __name__ =='__main__':
    main()