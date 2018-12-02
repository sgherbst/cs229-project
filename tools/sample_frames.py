import cv2
import numpy as np
import os.path
import os
import pickle
from random import random

from cs229.files import open_image, open_video, image_folder
from cs229.display import open_window, show_square_image, make_trackbar, get_trackbar
from cs229.image import img_to_mask
from cs229.util import FpsMon

def main(rate = 0.003, max_count=100):
    cap, props = open_video('reduced')

    folder = image_folder()

    count = 0
    while count < max_count:
        ok, img = cap.read()
        if not ok:
            break

        if random() > rate:
            continue

        count += 1
        path = os.path.join(folder, '{}.bmp'.format(count))

        cv2.imwrite(path, img)

if __name__ == "__main__":
    main()