import numpy as np
from scipy.stats import truncnorm

NORTH = np.array([0., 1.], dtype=float)


class Env(object):
    """Class for representing the environment.

    Contains a vector field and methods that are used for its visualization

    Parameters
    ----------
        vector_field: object
            Callable object which returns the values of the vector filed at
            given points when its `__call__` method is invoked.
        time_delta: float
            Lenght of the unit time increment.
        visuals_init: dict or None
            Dictionary containing the parameters for intialisation of the
            visualization (or None if visualization is not used).
    """
    def __init__(self, vector_field, time_delta, visuals_init=None):
        self.vector_field = vector_field
        self.time_delta = time_delta
        self.time_now = 0

        if visuals_init is not None:
            self.wraparoundmap = WrapAroundMap(**visuals_init['values'])
            sampler = TubeSampler(**visuals_init['sampler_init'])
            self.dots = sampler(visuals_init['num_dots'])
            self.use_visuals = True

        else:
            self.use_visuals = False

    def evaluate(self, points):
        """Method for evaluating the vector field at given points.

        Finds the values of the the vector field in the given points and stacks
        them into a array of shape (num_points, 2).

        Parameters
        ----------
            points: numpy.ndarray (dtype: float)
                Array of shape (num_points, 2) containing the points.

        Returns
        -------
            vectors: numpy.ndarray (dtype: float)
                Array of shape (num_points, 2) containing the vectors.
        """

        t = self.time_now
        f = lambda x: self.vector_field(x, t)
        vectors = np.apply_along_axis(f, 1, points)
        self.time_now += self.time_delta

        return vectors

    def visualize(self):
        """Method for calculating the new positions of visualization points.

        Moves the visualization points according to the flow of the vector
        field.

        Returns
        -------
            self.dots: numpy.ndarray (dtype: float)
                Array of shape (num_points, 2) containing the visualization
                points.
        """
        vectors = self.evaluate(self.dots)
        diff = vectors*self.time_delta
        self.dots = self.wraparoundmap(self.dots + diff)

        return self.dots


class FlowTube(object):
    """Class for representing a simple tubular flow.

    Model of a vector field representing a flow towards given direction
    (currently only positive y-axis is supported).

    Parameters
    ----------
        flow_map: object
            Callable object giving the initial values of the vector field.
        fluctuation: object
            Callable object for the vector fields strenght variations in
            time.
        center_point: float
            value of tubes center x-coordinate.
        direction: numpy.ndarray (dtype: float)
            Array of shape (2,). The direction of the flow (currently only
            [0.0, 1.0] is supported).
        """

    def __init__(self, flow_map, fluctuation, center_point, direction=NORTH):
        self.flow_map = flow_map
        self.fluctuation = fluctuation
        self.center_point = center_point
        self.direction = direction
        # currently only one direction is supported
        if not np.array_equal(direction, NORTH):
            raise NotImplementedError

    def __call__(self, point, time):
        """Evaluates the vector field at a given point."""
        strenght = self.flow_map(point)*self.fluctuation(time)
        vector = strenght*self.direction

        return vector


class BumpMap(object):
    """Class for representing a bump function."""
    def __init__(self, center, width):

        self.center = center
        self.width = width

    def __call__(self, num_float):
        """Calculate the bump function value for a given float."""
        r2 = (num_float - self.center)**2
        rad_squared = (self.width*0.5)**2

        if (r2 < rad_squared):
            return np.exp(1/rad_squared)*np.exp(-1/(rad_squared - r2))

        else:
            return 0.


class StaticUpFlow(FlowTube):
    """Static bump function valued flow towards postive y-direction."""
    def __init__(self, center, width, mid_value):
        bump_map = BumpMap(center, width)
        bump_func = lambda p: bump_map(p[0])
        static_map = lambda t: mid_value
        super().__init__(bump_func, static_map, center)


class TubeSampler(object):
    """Class for sampling points from regtangular area.

    This class is used for sampling points used for StaticUpFlow visualization.
    The x-coordinates are sampled from truncated normal distribution and the
    y-coordinate from uniform distribution.

    Parameters
    ----------
        x_range: list ([float, float])
            List with minimum and and maximum values of the sampling range.
        y_range:  list ([float, float])
            List with minimum and and maximum values of the sampling range.
    """
    def __init__(self, x_range, y_range):
        self.tr = truncnorm(x_range[0], x_range[1])
        self.y_vals = y_range

    def __call__(self, num_points):
        """Method for sampling points for the visualization.

        Selects the x-coordinates from truncated normal disribution and
        y-coordinates from uniform distribution, both supported on the
        intervals set in the initialization.

        Parameters
        ----------
            num_points: int
                number of points to sample.

        Returns
        -------
            points:  numpy.ndarray (dtype: float)
                Array of shape (num_points, 2) containing the points.
        """
        xs = self.tr.rvs((num_points, 1))
        ys = np.random.uniform(self.y_vals[0], self.y_vals[1], (num_points, 1))
        points = np.concatenate([xs, ys], axis=1)

        return points


class WrapAroundMap(object):
    """Class for constructing a function which wraps around the boundaries.

    This class is used for sending the visualization points which are about
    to escape from the boundary to opposite side of the regtangle (ie. the
    image is wrap around like a torus). For example if x is greater than the
    maximum value of x, it will be mapped to the minimum value of x, and vice
    versa. Same holds for y.

    parameters
    ----------
        min_x: float
            Minimum value of x.
        max_x: float
            Maximum value of x.
        min_y: float
            Minimum value of y.
        max_y: float
            Minimum value of y.
    """
    def __init__(self, min_x, max_x, min_y, max_y):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

    def __call__(self, array):
        """Applies the wrap around mapping to an array of vectors."""
        min_array = np.array([[self.min_x, self.min_y]], dtype=float)
        upper_bounds = (np.array([[self.max_x, self.max_y]], dtype=float)
            - min_array)

        return min_array + np.remainder(array - min_array, upper_bounds)
