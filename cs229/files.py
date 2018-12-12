import os
import os.path
import cv2

class CapProps:
    def __init__(self, width, height, fps):
        self.width=width
        self.height=height
        self.fps = fps

    @property
    def t_ms(self):
        return int(round(1e3/self.fps))

def get_dir(*args, mkdir_p=True):
    """
    Returns the absolute path to the relative directory specified by arg0, arg1, etc.
    """

    path_to_this_file = os.path.realpath(os.path.expanduser(__file__))
    top_dir = os.path.dirname(os.path.dirname(path_to_this_file))

    # find path to directory
    join_args = [top_dir] + [arg for arg in args]
    path = os.path.join(*join_args)

    # make the directory if needed/desired
    if mkdir_p:
        os.makedirs(path, exist_ok=True)

    # return the path
    return path

def get_file(*args, mkdir_p=False):
    # get path to the directory (and make directory if needed/desired)
    dir = get_dir(*args[:-1], mkdir_p=mkdir_p)

    # construct the path to the file
    path = os.path.join(dir, args[-1])

    return path

def read_video(name):
    file_path = get_file('input', 'videos', name)

    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    props = CapProps(width=width, height=height, fps=fps)

    return cap, props

def write_video(name, props):
    file_path = get_file('output', 'videos', name)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(file_path, fourcc, props.fps, (props.width, props.height))

    return writer