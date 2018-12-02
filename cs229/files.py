import os
import os.path
import pickle
import cv2
from tqdm import tqdm
import datetime

class CapProps:
    def __init__(self, width, height, fps):
        self.width=width
        self.height=height
        self.fps = fps

    @property
    def t_ms(self):
        return int(round(1e3/self.fps))

def top_dir():
    path_to_this_file = os.path.realpath(os.path.expanduser(__file__))
    return os.path.dirname(os.path.dirname(path_to_this_file))

def open_image(name=None):
    if name is None:
        name = 'test1'

    file_path = os.path.join(top_dir(), 'images', name + '.png')

    return cv2.imread(file_path, 0)

def open_video(name=None):
    if name is None:
        name = 'cropped'

    file_path = os.path.join(top_dir(), 'video', name + '.mp4')

    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    props = CapProps(width=width, height=height, fps=fps)

    return cap, props

def image_folder(name=None):
    if name is None:
        name = datetime.datetime.now().strftime('%m-%d_%H-%M-%S')

    path = os.path.join(top_dir(), 'images', name)
    os.makedirs(path)

    return path

def pickle_path(name):
    return os.path.join(top_dir(), 'pickles', name+'.p')

def write_pickle(name, data):
    pickle.dump(data, open(pickle_path(name), 'wb'))

def read_pickle(name):
    return pickle.load(open(pickle_path(name), 'rb'))

def fast_forward(cap, frames):
    for _ in tqdm(range(frames)):
        cap.read()