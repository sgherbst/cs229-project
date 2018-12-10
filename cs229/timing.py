from time import perf_counter
import numpy as np

class Profiler:
    def __init__(self):
        self.items = {}

    def tick(self, name):
        t = perf_counter()

        if name not in self.items:
            self.items[name] = {'last': None, 'deltas': []}

        self.items[name]['last'] = t

    def tock(self, name):
        t = perf_counter()

        delta = t - self.items[name]['last']
        self.items[name]['deltas'].append(delta)

    def stop(self):
        for name in self.items.keys():
            print('{}: {:0.1f} ms'.format(name, 1e3*np.mean(self.items[name]['deltas'])))