import numpy as np

NORTH = np.array([0., 1.], dtype=float)
ORIGIN = np.array([0., 0.], dtype=float)

class Env(object):

    def __init__(self, vector_field, update_map, time_delta):

        self.vector_field = vector_field
        self.time_delta = time_delta
        self.time_now = 0


    def evaluate(self, points):

        t = self.time_now
        f = lambda x: vector_field(x, t)
        vectors = np.apply_along_axis(f, 1, points)
        self.time_now += self.time_delta

        return vectors


    def visualize(self):

        raise NotImplementedError



class VectorField(object):

    def __init__(self, value_map):

        self.value_map = value_map


    def __call__(self, point, time):

        return value_map(point, time)



class FlowTube(object):

    def __init__(self, flow_map, fluctuation, center=ORIGIN, direction=NORTH):

        self.flow_map = flow_map
        self.fluctuation = fluctuation
        self.center = center
        self.direction = direction

        if direction != NORTH:
            raise NotImplementedError


    def __call__(point, time):

        strenght = self.flow_map(point)*self.fluctuation(time)
        vector = strenght*self.direction

        return vector



class BumpMap(object):

    def __init__(self, center, width):

        self.center = center
        self.width = width

    def __call__(num_float):

        r2 = (num_float - self.center)**2
        rad_squared = (self.width*0.5)**2

        if (r2 < rad_squared):
            return np.exp(1/rad_squared)*np.exp(1/(r2 - rad_squared))

        else:
            return 0.



class StaticUpFlow(FlowTube):

    def __init__(self, center, width, mid_value):

        bump_map = BumpMap(center, width)
        static_map = lambda t: mid_value
        super().__init__(flow_map, static_map)
