import numpy as np


class Env(object):

    def __init__(self, vector_field, update_map, time_delta):

        self.vector_field = vector_field
        self.update_map = update_map
        self.time_delta = time_delta


    def _update(self):

        update = self.update_map(self.vector_field, self.time_delta)
        self.vector_field += update


    def evaluate(self, points):

        self._update()
        vectors = np.apply_along_axis(lambda x: self.vector_field(x), 1, points)

        return vectors


    def visualize(self):

        raise NotImplementedError



class VectorField(object):

    def __init__(self, arg):

        self.arg = arg


    def __add__(self, other):

        raise NotImplementedError


    def __call__(self, point):

        raise NotImplementedError



class UpdateMap(object):

    def __init__(self, arg):

        self.arg = arg


    def __call__(self, vector_field, time_delta):

        raise NotImplementedError
