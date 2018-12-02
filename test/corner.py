import cv2
import numpy as np
import os.path
import os
import pickle

from cs229.files import read_pickle, write_pickle, open_video, fast_forward
from cs229.display import open_window, show_square_image
from cs229.image import circle_to_mask, crop_to_circle
from cs229.util import FpsMon

def main():
    cap, props = open_video('test')
    mouse_data = open_window(log_clicks=False)

    roi_circle = read_pickle('roi_circle')
    print(roi_circle)

    mask = circle_to_mask(roi_circle)
    mon = FpsMon()

    while (cap.isOpened()):
        mon.tick()

        ok, frame = cap.read()
        if not ok:
            break

        frame = crop_to_circle(frame, roi_circle)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.cornerHarris(gray, 2, 3, 0.04)

        out = frame.copy()
        corners2 = cv2.dilate(corners, None, iterations=3)
        out[corners2>0.01*corners2.max()] = [255, 0, 0]

        show_square_image(out)

        key = cv2.waitKey(props.t_ms)

        if key == ord('q'):
            break

    mon.done()

if __name__ == "__main__":
    main()