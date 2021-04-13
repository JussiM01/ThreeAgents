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


def rotate(vector, angle):
    rotation_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    return rotation_matrix.dot(vector)


def rotate_all(points, angle):
    return np.apply_along_axis(lambda x: rotate(x, angle), 1, points)


# --- Helpers for animation.py ------------------------------------------------

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
    fig = plt.figure(figsize=(params['fig_width'], params['fig_hight']))
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
