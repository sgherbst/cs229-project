import cv2
import numpy as np
import os.path
import os
import pickle

from cs229.files import read_pickle, write_pickle, open_video
from cs229.display import open_window, window_name
from cs229.image import mask_roi_circle

def make_circle(cx, cy, cr):
    return ((int(round(cx)), int(round(cy))), int(round(cr)))

def main(resize=4):
    cap, props = open_video()
    mouse_data = open_window(log_clicks=True)

    roi_circle = None

    while (cap.isOpened()):
        ret, frame = cap.read()

        # draw the points
        for click in mouse_data.clicks:
            click = click.scale(resize)
            cv2.circle(frame, click.to_tuple(), 10, (0, 255, 0), -1)

        # calculate the circle
        if len(mouse_data.clicks) >= 3:
            contour = np.array([[click.scale(resize).to_tuple()] for click in mouse_data.clicks])
            (cx, cy), cr = cv2.minEnclosingCircle(contour)
            roi_circle = make_circle(cx, cy, cr)
            mouse_data.clicks = []

        # apply the mask the mask
        if roi_circle is not None:
            frame = mask_roi_circle(frame, roi_circle)

        frame=frame[::resize, ::resize, :].copy()
        cv2.imshow(window_name(), frame)

        key = cv2.waitKey(props.t_ms)

        if key == ord('q'):
            break
        elif key == ord('c'):
            roi_circle = None
        elif key == ord('s'):
            write_pickle('roi_circle', roi_circle)
            print('Wrote ROI to pickle.')
        elif key == ord('l'):
            try:
                roi_circle = read_pickle('roi_circle')
                print('Read ROI from pickle.')
            except OSError:
                print('Could not open ROI circle file.')

if __name__ == "__main__":
    main()