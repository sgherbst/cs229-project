import cv2

from cs229.files import open_video
from cs229.display import open_window, show_image, make_trackbar, get_trackbar
from cs229.image import img_to_mask, erode
import numpy as np

def main():
    # load first frame from video
    cap, props = open_video('cropped')
    ok, img = cap.read()
    img = img[:, :, 0]

    arr = np.zeros(img.shape, dtype=float)
    count = 0

    while True:
        # read image
        ok, img = cap.read()
        if not ok:
            break

        img = img[:, :, 0].astype(float)

        arr += img
        count += 1

        if count % 100 == 0:
            print('Frame {}'.format(count))

    arr /= count
    arr = np.clip(np.round(arr), 0, 255).astype(np.uint8)
    cv2.imwrite('background.bmp', arr)

if __name__ == "__main__":
    main()