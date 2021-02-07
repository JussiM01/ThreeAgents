import numpy as np

NORTH = np.array([0., 1.], dtype=float)


class Env(object):

    def __init__(self, vector_field, time_delta):

        self.vector_field = vector_field
        self.time_delta = time_delta
        self.time_now = 0


    def evaluate(self, points):

        t = self.time_now
        f = lambda x: self.vector_field(x, t)
        vectors = np.apply_along_axis(f, 1, points)
        self.time_now += self.time_delta

        return vectors


    def visualize(self):

        raise NotImplementedError



class FlowTube(object):

    def __init__(self, flow_map, fluctuation, center_point, direction=NORTH):

        self.flow_map = flow_map
        self.fluctuation = fluctuation
        self.center_point = center_point
        self.direction = direction

        if not np.array_equal(direction, NORTH):
            raise NotImplementedError


    def __call__(self, point, time):

        strenght = self.flow_map(point)*self.fluctuation(time)
        vector = strenght*self.direction

        return vector



class BumpMap(object):

    def __init__(self, center, width):

        self.center = center
        self.width = width

    def __call__(self, num_float):

        r2 = (num_float - self.center)**2
        rad_squared = (self.width*0.5)**2

        if (r2 < rad_squared):
            return np.exp(1/rad_squared)*np.exp(1/(r2 - rad_squared))

        else:
            return 0.



class StaticUpFlow(FlowTube):

    def __init__(self, center, width, mid_value):

        bump_map = BumpMap(center, width)
        bump_func = lambda p: bump_map(p[0])
        static_map = lambda t: mid_value
        super().__init__(bump_func, static_map, center)
