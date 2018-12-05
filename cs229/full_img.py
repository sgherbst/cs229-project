import cv2
import os
import os.path
import matplotlib.pyplot as plt

from cs229.files import top_dir
from cs229.image import img_to_mask

class FullImage:
    def __init__(self, img, mask=None, thresh=115, fly_color='black'):
        # convert to grayscale if necessary
        if len(img.shape) == 3:
            img = img[:, :, 0]

        # make sure image is now grayscale
        assert len(img.shape) == 2

        # save image
        self.img = img

        # save mask
        self.mask = mask

        # determine thresholding type
        if fly_color.lower() == 'black':
            self.thresh_type = cv2.THRESH_BINARY_INV
        elif fly_color.lower() == 'white':
            self.thresh_type = cv2.THRESH_BINARY
        else:
            raise Exception('Invalid fly color.')

        # threshold image
        self.thresh = thresh
        _, bw = cv2.threshold(img, self.thresh, 255, self.thresh_type)

        # apply mask to image if desired
        if mask is not None:
            bw = cv2.bitwise_and(bw, mask)

        # save black and white image
        self.bw = bw

        # extract contours
        _, contours, _ = cv2.findContours(self.bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # save contours
        self.contours = contours

def main():
    image_path = os.path.join(top_dir(), 'images', '12-01_10-41-12', '2.bmp')
    img = cv2.imread(image_path)

    mask = img_to_mask(img)
    full_img = FullImage(img, mask=mask)

    print(len(full_img.contours))

    plt.imshow(full_img.img)
    plt.show()

if __name__ == "__main__":
    main()

# def main():
#     mouse_data = open_window(log_clicks=True)
#
#     test = None
#     old_test = None
#
#     img = None
#
#     cap, props = open_video('cropped')
#     ok, img = cap.read()
#     img = img[:, :, 0]
#     img = cv2.resize(img, (512, 512))
#     dim = img.shape[0]
#     cmask = img_to_mask(img)
#
#     while True:
#         ok, img = cap.read()
#         if not ok:
#             break
#         img = img[:, :, 0]
#
#         tick = perf_counter()
#         img = cv2.resize(img, (512, 512))
#         tock = perf_counter()
#
#         _, tmask = cv2.threshold(img, 115, 255, cv2.THRESH_BINARY_INV)
#         fmask = cv2.bitwise_and(cmask, tmask)
#
#         im2, contours, hierarchy = cv2.findContours(fmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#
#         contours = sorted(contours, key=lambda cnt: -cv2.contourArea(cnt))
#         contour = contours[0]
#
#         omask = np.zeros((dim, dim), dtype=np.uint8)
#         cv2.drawContours(omask, [contour], 0, (255, 255, 255), -1)
#
#
#         xmin = np.min(contour[:, 0, 1])
#         xmax = np.max(contour[:, 0, 1])
#         ymin = np.min(contour[:, 0, 0])
#         ymax = np.max(contour[:, 0, 0])
#
#         out = cv2.bitwise_and(img, img, mask=omask)
#         out = out[xmin:xmax+1, ymin:ymax+1]
#
#         M = cv2.moments(out)
#
#
#         print('Processing time: {} ms'.format((tock-tick)*1e3))
#
#         show_square_image(out)
#
#         key = cv2.waitKey(25)
#
#         if key == ord('q'):
#             break