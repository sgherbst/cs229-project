from timeit import default_timer as timer

class TickTock:
    def __init__(self, name=None, update_interval=1):
        self.update_interval = update_interval
        self.name = name

        self.start_time = None
        self.last_update_time = None

        self.count = 0
        self.elapsed = 0

    def tick(self):
        t = timer()
        self.start_time = t

    def tock(self):
        t = timer()

        self.elapsed += t - self.start_time
        self.count += 1

        # display update
        if self.update_interval is not None:
            if self.last_update_time is None:
                self.last_update_time = t

            time_since_update = t - self.last_update_time
            if time_since_update >= self.update_interval:
                prepend = self.name + ' ' if self.name is not None else ''
                print('{}Average: {:0.3f} ms'.format(prepend, self.elapsed/self.count*1e3))
                self.last_update_time = t
                self.count = 0
                self.elapsed = 0

class FpsMon:
    def __init__(self, update_interval=1):
        self.update_interval = update_interval

        self.start_time = None
        self.last_update_time = None

        self.update_frame_count = 0
        self.total_frame_count = 0

    def tick(self):
        # get the current time
        t = timer()

        # if this is the first frame, save the time
        if self.start_time is None:
            self.start_time = t

        # display FPS update if enabled
        if self.update_interval is not None:
            if self.last_update_time is None:
                self.last_update_time = t

            dt = t - self.last_update_time
            if dt >= self.update_interval:
                print('FPS: {:0.3f}'.format(self.update_frame_count/dt))
                self.last_update_time = t
                self.update_frame_count = 0

        # update frame counters
        self.update_frame_count += 1
        self.total_frame_count += 1

    def done(self):
        t = timer()

        try:
            dt = t - self.start_time
            dt_str = '{:0.3f} s'.format(dt)
        except:
            dt = None
            dt_str = 'N/A'

        try:
            fps = self.total_frame_count / dt
            fps_str = '{:0.3f}'.format(fps)
        except:
            fps_str = 'N/A'

        print('*** Summmary ***')
        print('Total frames: {}'.format(self.total_frame_count))
        print('Total duration: ' + dt_str)
        print('Average FPS: ' + fps_str)
