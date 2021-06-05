import json
import os

import matplotlib.pyplot as plt
import numpy as np


# --- Helpers for main.py -----------------------------------------------------

def load_config(filename):
    """Loads the configuration file."""
    config_file = os.path.join('config_files',  filename)
    with open(config_file, 'r') as conf_file:
        config = json.load(conf_file)

    return config


def random_intial_positions(anim_dict, parsed_args):
    """Intializes the agents' positions randomly according to a given range."""
    x_axis_len = anim_dict['ax_x_max'] - anim_dict['ax_x_min']
    y_axis_len = anim_dict['ax_y_max'] - anim_dict['ax_y_min']

    # set the starting ranges for the agents intial positions
    start_min_x = anim_dict['ax_x_min'] + x_axis_len*parsed_args.x_min
    start_max_x = anim_dict['ax_x_min'] + x_axis_len*parsed_args.x_max
    start_min_y = anim_dict['ax_y_min'] + y_axis_len*parsed_args.y_min
    start_max_y = anim_dict['ax_y_min'] + y_axis_len*parsed_args.y_max

    # choose the agents intial positions randomly from these ranges
    points_x = np.random.uniform(start_min_x, start_max_x, (3, 1))
    points_y = np.random.uniform(start_min_y, start_max_y, (3, 1))
    initial_positions = np.concatenate([points_x, points_y], axis=1)

    return initial_positions


# --- Helpers for interactionmodels.py ----------------------------------------

def conjugate_product(vector1, vector2):
    """Multiplies as complex numbers vector1 and conjugate of vector2."""
    vec1_complex = np.array([vector1[0] + 1j*vector1[1]])
    vec2_complex = np.array([vector2[0] + 1j*vector2[1]])

    return vec1_complex*vec2_complex.conj()


def normalize(vector):
    """Returns vector divided by its norm (or the vector if zero)."""
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector/norm


def normalize_all(vectors):
    """Returns normalized vectors."""
    return np.apply_along_axis(lambda x: normalize(x), 1, vectors)


def rotate(vector, angle):
    """Rotates a vector by given angle."""
    rotation_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    return rotation_matrix.dot(vector)


def rotate_all(points, angle):
    """Rotates array of vectors by given angle."""
    return np.apply_along_axis(lambda x: rotate(x, angle), 1, points)


# --- Helpers for animation.py ------------------------------------------------

class RandomTopography:
    """Class for creating random countors which resemble a topograpy map.

    This class is used for sampling positive and negative gaussian bumps which
    are added together for getting the height values of a random topograpy.
    These are used for creating the matplotlib countors for the animation plot.

    Parameters
    ----------
        random_scales: {
                        'ax_x_min': float,
                        'ax_x_max': float,
                        'ax_y_min': float,
                        'ax_y_max': float,
                        'cov_diag_min': float,
                        'cov_diag_max': float,
                        'cov_offd_min': float,
                        'cov_offd_max': float
                        }
            Dictionary containing the values from which the parameters of the
            gaussian distributions are drawn.
        countor_params: {
                        'num_x_grid': int,
                        'num_y_grid': int,
                        'colors': str
                        }
            Dictionary containing the countors grid sizes and colors string.
        num_gauss: int
            Number of postive and negative gaussian bumps (same used for both).

    """

    def __init__(self, random_scales, num_gauss, countor_params):
        self.mean_x_min = random_scales['ax_x_min']
        self.mean_x_max = random_scales['ax_x_max']
        self.mean_y_min = random_scales['ax_x_min']
        self.mean_y_max = random_scales['ax_x_max']
        self.cov_diag_min = random_scales['cov_diag_min']
        self.cov_diag_max = random_scales['cov_diag_max']
        self.cov_offd_min = random_scales['cov_offd_min']
        self.cov_offd_max = random_scales['cov_offd_max']
        self.num_gauss = num_gauss
        self.num_x_grid = countor_params['num_x_grid']
        self.num_y_grid = countor_params['num_y_grid']
        self.colors = countor_params['colors']

    def _sample_params(self):
        means = []
        covs = []

        for _ in range(self.num_gauss):
            mean_x = np.random.uniform(self.mean_x_min, self.mean_x_max)
            mean_y = np.random.uniform(self.mean_x_min, self.mean_x_max)

            cov_xx = np.random.uniform(self.cov_diag_min, self.cov_diag_max)
            cov_xy = np.random.uniform(self.cov_offd_min, self.cov_offd_max)
            cov_yy = np.random.uniform(self.cov_diag_min, self.cov_diag_max)

            means.append([mean_x, mean_y])
            covs.append([[cov_xx, cov_xy], [cov_xy, cov_yy]])

        return means, covs

    def _sample_heights(self, x_grid, y_grid):
        pos_means, pos_covs = self._sample_params()
        neg_means, neg_covs = self._sample_params()

        pos = []
        neg = []

        arr = np.stack([x_grid, y_grid], axis=2)
        rv = multivariate_normal()

        for i in range(self.num_gauss):
            rv_pos = multivariate_normal(pos_means[i], pos_covs[i])
            rv_neg = multivariate_normal(neg_means[i], neg_covs[i])
            z_pos = rv_pos.pdf(arr)
            z_neg = rv_neg.pdf(arr)

            pos.append(z_pos)
            neg.append(z_neg)

        pos_values = np.sum(np.stack(pos, axis=0), axis=0)
        neg_values = np.sum(np.stack(neg, axis=0), axis=0)

        return pos_values - neg_values

    def create_countors(self, axes):
        """Creates the random countors to a given axes object."""
        x = np.linspace(ax_x_min, ax_x_max, self.num_x_grid)
        y = np.linspace(ax_y_min, ax_y_max, self.num_y_grid)

        X, Y = np.meshgrid(x, y)
        Z = self._sample_heights(X, Y)

        plt.rcParams['contour.negative_linestyle'] = 'solid'
        countors = axes.contour(X, Y, Z, colors=self.colors)

        return countors


def init_scatter(params, axes, points):
    """Function for intializing a scatter artist.

    Sets the sizes and colors of the points that the scatter artist will
    be handling (points for the agents or the visualization dots).

    Parameters
    ----------
        params: dict
            Dictionary which contains parameters for the sizes and colors.
        axes: matplotlib.axes.Axes
            Axes object of the figure used for the animation.

    Returns
    -------
        scatter: matplotlib.axes.collections.PathCollection
            Scatter artist handling the plotting of the given points.

    """
    size = np.array((params['pointsize']), dtype=float)
    sizes = np.repeat((size), points.shape[0], axis=0)
    facecolor = np.array([params['facecolors']], dtype=float)
    facecolors = np.repeat(facecolor, points.shape[0], axis=0)
    edgecolor = np.array([params['edgecolors']], dtype=float)
    edgecolors = np.repeat(edgecolor, points.shape[0], axis=0)
    scatter = axes.scatter(
        points[:, 0], points[:, 1], s=sizes, lw=0.5, facecolors=facecolors,
        edgecolors=edgecolors)

    return scatter


def init_animation(params, points, dots=None):
    """Function for intializing the figure and artists for the animation.

    Sets the figure according to given parameters and intializes the artists
    that will be drawing to it.

    Parameters
    ----------
        params: dict
            Dictionary which contains parameters for the figure and artists.
        points:
        dots:

    Returns
    -------
        if dots are None:
            (fig, scatter): (
                matplotlib.figure.Figure,
                matplotlib.axes.collections.PathCollection
                )
            Tuple with the figure and the scatter artist for the agents.
        else:
            (fig, env_scatter, scatter): (
                matplotlib.figure.Figure,
                matplotlib.axes.collections.PathCollection
                )
            Tuple with the figure and the scatter artists for both the agents
            and the dots.

    """
    fig = plt.figure(figsize=(params['fig_width'], params['fig_height']))
    axes = fig.add_axes(
        [params['x_min'], params['y_min'], params['x_max'], params['y_max']],
        frameon=params['frameon'])
    axes.set_xlim(params['ax_x_min'], params['ax_x_max'])
    axes.set_ylim(params['ax_y_min'], params['ax_y_max'])

    if params['remove_thicks']:
        axes.set_xticks([])
        axes.set_yticks([])

    axes.grid(params['use_grid'])
    scatter = init_scatter(params, axes, points)

    if dots is not None:
        env_scatter_params = {
            'pointsize': 0.025,
            'edgecolors': [1, 0, 0, 1],
            'facecolors': [1, 0, 0, 1]
            }
        env_scatter = init_scatter(env_scatter_params, axes, dots)

        return fig, env_scatter, scatter

    return fig, scatter
