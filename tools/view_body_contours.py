import cv2
import numpy as np
import matplotlib.pyplot as plt

from cs229.files import open_video
from cs229.display import open_window, show_image, make_trackbar, get_trackbar
from cs229.image import img_to_mask, erode
from cs229.contour import threshold_wings, find_wing_contours

from time import perf_counter

def main():
    # load first frame from video
    cap, props = open_video('test4')

    # open the window and show the first image
    open_window()

    while True:
        # read image
        ok, img = cap.read()
        if not ok:
            break
        img = img[:, :, 0]
        mask = img_to_mask(img)

        # generate mask
        tick = perf_counter()
        contours = find_wing_contours(img, mask)
        tock = perf_counter()
        print(1e3*(tock-tick))

        out = np.zeros(img.shape, dtype=np.uint8)
        cv2.drawContours(out, contours, -1, (255, 255, 255), -1)

        # show result
        show_image(out, downsamp=2)

        key = cv2.waitKey(props.t_ms)
        if key == ord('q'):
            break

if __name__ == "__main__":
    main()