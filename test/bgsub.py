import numpy as np
import cv2 as cv

from cs229.files import read_pickle, write_pickle, open_video, fast_forward
from cs229.display import open_window, show_square_image
from cs229.image import circle_to_mask, crop_to_circle
from cs229.util import FpsMon

def main():
    cap, props = open_video('test1')


    #fgbg = cv.bgsegm.createBackgroundSubtractorMOG()
    #fgbg = cv.createBackgroundSubtractorMOG2()
    fgbg = cv.bgsegm.createBackgroundSubtractorGMG()
    mon = FpsMon()

    while(1):
        mon.tick()

        ok, frame = cap.read()
        if not ok:
            return

        frame = cv.resize(frame, (512, 512))
        fgmask = fgbg.apply(frame)

        cv.imshow('frame',fgmask)
        k = cv.waitKey(30) & 0xff
        if k == 27:
           break

    mon.done()

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()