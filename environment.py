import numpy as np
from scipy.stats import truncnorm

NORTH = np.array([0., 1.], dtype=float)


class Env(object):

    def __init__(self, vector_field, time_delta, visuals_init=None):

        self.vector_field = vector_field
        self.time_delta = time_delta
        self.time_now = 0

        if visuals_init is not None:
            self.wraparoundmap = WrapAroundMap(**visuals_init['values'])
            self.dots = TubeSampler(**visuals_init['sampler_args'])


    def evaluate(self, points):

        t = self.time_now
        f = lambda x: self.vector_field(x, t)
        vectors = np.apply_along_axis(f, 1, points)
        self.time_now += self.time_delta

        return vectors


    def visualize(self):

        vectors = self.evaluate(self.dots)
        diff = vectors*self.time_delta
        self.dots = self.wraparoundmap(self.dots + diff)

        return self.dots



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
            return np.exp(1/rad_squared)*np.exp(-1/(rad_squared - r2))

        else:
            return 0.



class StaticUpFlow(FlowTube):

    def __init__(self, center, width, mid_value):

        bump_map = BumpMap(center, width)
        bump_func = lambda p: bump_map(p[0])
        static_map = lambda t: mid_value
        super().__init__(bump_func, static_map, center)



class TubeSampler(object):

    def __init__(self, x_range, y_range):

        self.tr = truncnorm(x_range[0], x_range[1])

    def __call__(self, num_points):

        xs = self.tr.rvs((num_points, 1))
        ys = np.random.uniform(y_range[0], y_range[1], (num_points, 1))
        points = np.concatenate([xs, ys], axis=1)

        return points



class WrapAroundMap(object):

    def __init__(self, min_x, max_x, min_y, max_y):

        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

    def __call__(self, array):

        min_array = np.array([[self.min_x, self.min_y]], dtype=float)
        upper_bounds = (np.array([[self.max_x, self.max_y]], dtype=float) -
            min_array)

        return min_array + np.remainder(array - min_array, upper_bounds)
