import os
import os.path
import pickle
import cv2
from tqdm import tqdm
import datetime
from glob import glob
from itertools import chain

class CapProps:
    def __init__(self, width, height, fps):
        self.width=width
        self.height=height
        self.fps = fps

    @property
    def t_ms(self):
        return int(round(1e3/self.fps))

def get_annotation_files():
    folders = ['12-04_17-54-43', '12-05-12-43-00', '12-07_16_45_00', '12-08_11-15-00', '12-08_22_00_00']
    folders = [os.path.join(top_dir(), 'images', folder) for folder in folders]

    files = [glob(os.path.join(folder, '*.json')) for folder in folders]

    return chain(*files)

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