import cv2
import numpy as np
import os.path
import os
import pickle
from tqdm import tqdm

def main(win='roi', pickle_file='roi_circle.p', resize_dim=512):
    # open file
    file_dir = '/Users/sgherbst/Dropbox/Laptop Sync/Stanford/Classes/CS229/Project/Data/Video/'
    file_name = 'Basler acA2440-35um (22467982)_20180926_131727243.m4v'
    cap = cv2.VideoCapture(os.path.join(file_dir, file_name))

    # get properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # make the mask
    roi_circle = pickle.load(open(pickle_file, 'rb'))
    (cx, cy), cr = roi_circle
    mask = np.zeros((height, width), np.uint8)
    cv2.circle(mask, *roi_circle, 255, -1)

    # calculate delay time for each loop iteration in milliseconds
    t_loop = int(round(1e3 / fps))

    # ref: https://stackoverflow.com/questions/3002085/python-to-print-out-status-bar-and-percentage
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.bitwise_and(frame, frame, mask=mask)

        # crop image
        frame=frame[cy-cr:cy+cr, cx-cr:cx+cr, :].copy()
        frame=cv2.resize(frame, (resize_dim, resize_dim))

        # write image

        # display image
        cv2.imshow(win, frame)
        key = cv2.waitKey(t_loop) & 0xff
        if key == ord('q'):
            break

    # Release I/O
    cap.release()

    # Close frames
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()