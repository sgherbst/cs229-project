import cv2

class Click:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @staticmethod
    def scale_one(value, factor):
        return int(round(value*factor))

    def scale(self, factor):
        return Click(x=self.scale_one(self.x, factor), y=self.scale_one(self.y, factor))

    def to_tuple(self):
        return (self.x, self.y)

class MouseData:
    def __init__(self, log_clicks):
        self.x = 0
        self.y = 0
        self.clicks = []
        self.log_clicks = log_clicks

    def callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.log_clicks:
                self.clicks.append(Click(x, y))
        elif event == cv2.EVENT_MOUSEMOVE:
            self.x = x
            self.y = y

def window_name():
    return 'CS229'

def make_trackbar(name, default, max):
    def nothing(x):
        pass

    cv2.createTrackbar(name, window_name(), default, max, nothing)

def get_trackbar(name, min=0, fixed_point=None):
    value = cv2.getTrackbarPos(name, window_name())

    if fixed_point is not None:
        value *= fixed_point

    value = max(min, value)

    return value

def open_window(log_clicks=False):
    cv2.namedWindow(window_name())
    mouse_data = MouseData(log_clicks=log_clicks)
    cv2.setMouseCallback(window_name(), mouse_data.callback)
    return mouse_data

def square_disp_size():
    return 512

def show_square_image(frame):
    disp = cv2.resize(frame, (square_disp_size(), square_disp_size()))
    cv2.imshow(window_name(), disp)

def show_image(frame):
    cv2.imshow(window_name(), frame)
