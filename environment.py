import numpy as np

from scipy.stats import truncnorm

NORTH = np.array([0., 1.], dtype=float)


class Env:
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

    Attributes
    ----------
        vector_field: object
            Callable object which returns the values of the vector filed at
            given points when its `__call__` method is invoked.
        time_delta: float
            Lenght of the unit time increment.
        visuals_init: dict or None
            Dictionary containing the parameters for intialisation of the
            visualization (or None if visualization is not used).
        time_now: float
            Time from the start.
        use_visuals: bool
            False if `visuals_init` is None else True.
        wraparoundmap: object
            (Set only if `visuals_init` is not None.)
            Callable object responsible for stopping points from escaping at
            the boundary.
        sampler: TubeSampler
            (Set only if `visuals_init` is not None.)
            Samples the original postions of the points for flow visualization.
        dots: numpy.ndarray (dtype: float)
            (Set only if `visuals_init` is not None.)
            Positions of the visualization points.

    """

    def __init__(self, vector_field, time_delta, visuals_init=None):
        self.vector_field = vector_field
        self.time_delta = time_delta
        self.time_now = 0
        self.visuals_init = visuals_init

        if visuals_init is not None:
            self.wraparoundmap = WrapAroundMap(**visuals_init['values'])
            sampler = TubeSampler(**visuals_init['sampler_init'])
            self.dots = sampler(visuals_init['num_dots'])
            self.use_visuals = True

        else:
            self.use_visuals = False

    def __repr__(self):
        args = (self.vector_field, self.time_delta, self.visuals_init)
        return 'Env({}, {}, {}, {})'.format(*args)

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
        vectors = np.apply_along_axis(
            lambda x: self.vector_field(x, t), 1, points)
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


class FlowTube:
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
        self._flow_map = flow_map
        self._fluctuation = fluctuation
        self._center_point = center_point
        self._direction = direction
        # currently only one direction is supported
        if not np.array_equal(direction, NORTH):
            raise NotImplementedError

    def __repr__(self):
        args = (self._flow_map, self._fluctuation, self._center_point,
                self._direction)
        return 'FlowTube({}, {}, {}, {})'.format(*args)

    def __call__(self, point, time):
        """Evaluates the vector field at a given point."""
        strenght = self._flow_map(point)*self._fluctuation(time)
        vector = strenght*self._direction

        return vector


class BumpMap:
    """Class for representing a bump function.

    Creates a smooth exponential bump function corresponding to given
    center value and width.

    Parameters
    ----------
        center: float
            Center point where the function value is one.
        width: float
            For arguments with absolute value greater than half of this the
            function value is zero.

    """

    def __init__(self, center, width):
        self._center = center
        self._width = width

    def __repr__(self):
        return 'BumpMap({}, {})'.format(self._center, self._width)

    def __call__(self, num_float):
        """Calculate the bump function value for a given float."""
        dist_square = (num_float - self._center)**2
        rad_square = (self._width*0.5)**2

        if dist_square < rad_square:
            return np.exp(1/rad_square)*np.exp(-1/(rad_square - dist_square))

        return 0.


class StaticUpFlow(FlowTube):
    """Static bump function valued flow towards postive y-direction.

    Creates a FlowTube that is supported in a rectangular area with values
    decreasing according to a bump function valued distribution in towards
    x-direction at the tube boundary.

    Parameters
    ----------
        center: [float, float]
            Coordinates of the middle point.
        width: float
            width of the tube in the x-direction.
        mid_value: float
            Strenght of the flow in the vertical middle line.

    """

    def __init__(self, center, width, mid_value):
        bump_map = BumpMap(center, width)
        super().__init__(lambda p: bump_map(p[0]), lambda t: mid_value, center)
        self._center = center
        self._width = width
        self._mid_value = mid_value

    def __repr__(self):
        args = (self._center, self._width, self._mid_value)
        return 'StaticUpFlow({}, {}, {})'.format(*args)


class TubeSampler:
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
        self._truncnorm = truncnorm(x_range[0], x_range[1])
        self._y_vals = y_range

    def __repr__(self):
        return 'TubeSampler({}, {})'.format(self.x_range, self.y_range)

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
            points: numpy.ndarray (dtype: float)
                Array of shape (num_points, 2) containing the points.

        """
        x_coordinates = self._truncnorm.rvs((num_points, 1))
        y_coordinates = np.random.uniform(
            self._y_vals[0], self._y_vals[1], (num_points, 1))
        points = np.concatenate([x_coordinates, y_coordinates], axis=1)

        return points


class WrapAroundMap:
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
        self._min_x = min_x
        self._max_x = max_x
        self._min_y = min_y
        self._max_y = max_y

    def __repr__(self):
        args = (self._min_x, self._max_x, self._min_y, self._max_y)
        return 'WrapAroundMap({}, {}, {}, {})'.format(*args)

    def __call__(self, array):
        """Applies the wrap around mapping to an array of vectors."""
        min_array = np.array([[self._min_x, self._min_y]], dtype=float)
        upper_bounds = (np.array([[self._max_x, self._max_y]], dtype=float)
                        - min_array)

        return min_array + np.remainder(array - min_array, upper_bounds)
