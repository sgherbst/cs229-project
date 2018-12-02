import cv2
import numpy as np
import os.path
import os
import pickle
from tqdm import tqdm

def main(win='roi', pickle_file='roi_circle.p', resize_dim=128, should_display=False, should_write=True):
    # open file
    file_dir = '/Users/sgherbst/Dropbox/Laptop Sync/Stanford/Classes/CS229/Project/Data/Video/'
    file_name = 'Basler acA2440-35um (22467982)_20180926_131727243.m4v'
    cap = cv2.VideoCapture(os.path.join(file_dir, file_name))

    # get properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # fix length
    if length < 0:
        length = 30000

    # create writer
    if should_write:
        out = cv2.VideoWriter('cropped.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (resize_dim, resize_dim))

    # make the mask
    roi_circle = pickle.load(open(pickle_file, 'rb'))
    (cx, cy), cr = roi_circle
    mask = np.zeros((height, width), np.uint8)
    cv2.circle(mask, *roi_circle, 255, -1)

    # calculate delay time for each loop iteration in milliseconds
    if should_display:
        t_loop = int(round(1e3 / fps))

    # ref: https://stackoverflow.com/questions/3002085/python-to-print-out-status-bar-and-percentage
    count = 0
    with tqdm(total=length) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.bitwise_and(frame, frame, mask=mask)

            # crop image
            frame=frame[cy-cr:cy+cr, cx-cr:cx+cr, :].copy()
            frame=cv2.resize(frame, (resize_dim, resize_dim))

            # write image
            if should_write:
                out.write(frame)

            # display image
            if should_display:
                cv2.imshow(win, frame)
                key = cv2.waitKey(t_loop) & 0xff
                if key == ord('q'):
                    break

            count += 1
            pbar.update(1)

    # Release I/O
    cap.release()
    if should_write:
        out.release()

    # Close frames
    if should_display:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()