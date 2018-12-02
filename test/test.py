#ref: https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/
import cv2
import sys
import pickle
import os
import os.path
from time import time
import numpy as np

from cs229.files import read_pickle, write_pickle, open_video, fast_forward
from cs229.display import open_window, show_square_image
from cs229.image import make_mask, apply_mask, crop_to_roi

def make_analysis_frame(f):
    return cv2.resize(f, (128,128))

def make_display_frame(f):
    return cv2.resize(f, (512,512))

def main(use_pickle=False):
    # open video
    cap, props = open_video()

    # make the mask
    roi_circle = read_pickle('roi_circle')
    (cx, cy), cr = roi_circle
    mask = np.zeros((props.height, props.width), np.uint8)
    cv2.circle(mask, *roi_circle, 255, -1)

    #tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

    tracker = cv2.TrackerBoosting_create
    #tracker = cv2.TrackerMIL_create
    #tracker = cv2.TrackerKCF_create
    #tracker = cv2.TrackerTLD_create
    #tracker = cv2.TrackerMedianFlow_create
    #tracker = cv2.TrackerMOSSE_create
    #tracker = cv2.TrackerCSRT_create

    # Fast forward to a specific frame
    ok, frame = cap.read()
    frame = frame[cy - cr:cy + cr:4, cx - cr:cx + cr:4, :].copy()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    # Uncomment the line below to select a different bounding box
    trackers = cv2.MultiTracker_create()
    colors = [(255,0,0), (0,255,0)]

    for n, _ in enumerate(colors):
        print('Set bounding box for tracker {}.'.format(n))

        bbox = cv2.selectROI(frame, False)

        trackers.add(tracker(), frame, bbox)

    fps_frames = 0
    fps_time = 0

    while True:
        # Read a new frame
        ok, frame = cap.read()
        if not ok:
            break

        # Crop
        frame = frame[cy - cr:cy + cr:4, cx - cr:cx + cr:4, :].copy()

        # save frame for display
        disp_frame = frame.copy()

        tick = time()
        (success, boxes) = trackers.update(frame)
        fps_time += (time() - tick)

        for n, box in enumerate(boxes):
            # Tracking success
            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
            cv2.rectangle(disp_frame, p1, p2, colors[n], 2, 1)

        fps_frames += 1
        if fps_frames > 30:
            print('FPS: {}'.format(fps_frames/fps_time))
            fps_frames = 0
            fps_time = 0

        # Display tracker type on frame
        #cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display FPS on frame
        #cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display result
        cv2.imshow("Tracking", disp_frame)

        # Exit if ESC pressed
        k = cv2.waitKey(28) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
