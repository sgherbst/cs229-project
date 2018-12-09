import cv2

from cs229.files import open_video
from cs229.display import open_window, show_image, make_trackbar, get_trackbar
from cs229.image import img_to_mask, erode

def main():
    # load first frame from video
    cap, props = open_video('test4')
    ok, img = cap.read()
    img = img[:, :, 0]

    # open the window and show the first image
    open_window()
    show_image(img, downsamp=2)

    # add trackbars
    make_trackbar('min', 0, 255)
    make_trackbar('max', 213, 255)

    while True:
        # read image
        ok, img = cap.read()
        if not ok:
            break
        img = img[:, :, 0]

        # generate mask
        cmask = img_to_mask(img)

        fmask = cv2.inRange(img, get_trackbar('min'), get_trackbar('max'))
        mask = cv2.bitwise_and(cmask, fmask)

        out = cv2.bitwise_and(img, img, mask=mask)
        print(get_trackbar('max'))

        show_image(out, downsamp=2)

        key = cv2.waitKey(props.t_ms)
        if key == ord('q'):
            break

if __name__ == "__main__":
    main()