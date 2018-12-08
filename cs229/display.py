import cv2

WINDOW_NAME = 'CS229'

def nothing(x):
    pass

class MouseData:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.clicks = []

    def callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicks.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE:
            self.x = x
            self.y = y

    def last_click(self):
        try:
            return self.clicks[-1]
        except:
            return None

    def clear_clicks(self):
        self.clicks = []

def open_window(mouse_callback=False):
    cv2.namedWindow(WINDOW_NAME)

    if mouse_callback:
        mouse_data = MouseData()
        cv2.setMouseCallback(WINDOW_NAME, mouse_data.callback)

        return mouse_data
    else:
        return None

def show_image(frame, downsamp=1):
    frame = frame[::downsamp, ::downsamp]
    cv2.imshow(WINDOW_NAME, frame)

def make_trackbar(name, default, max):
    cv2.createTrackbar(name, WINDOW_NAME, default, max, nothing)

def get_trackbar(name):
    return cv2.getTrackbarPos(name, WINDOW_NAME)