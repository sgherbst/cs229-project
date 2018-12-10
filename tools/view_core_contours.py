import cv2
import numpy as np

from cs229.files import open_video
from cs229.display import open_window, show_image, make_trackbar, get_trackbar
from cs229.image import img_to_mask, erode
from cs229.contour import threshold_core, find_core_contours

def main():
    # load first frame from video
    cap, props = open_video('test1')

    # open the window and show the first image
    open_window()

    while True:
        # read image
        ok, img = cap.read()
        if not ok:
            break
        img = img[:, :, 0]

        # generate mask
        mask = img_to_mask(img)
        contours = find_core_contours(img, mask)

        # show result
        out = np.zeros(img.shape, dtype=np.uint8)
        cv2.drawContours(out, contours, -1, (255, 255, 255), -1)
        show_image(out, downsamp=2)

        key = cv2.waitKey(props.t_ms)
        if key == ord('q'):
            break

if __name__ == "__main__":
    main()