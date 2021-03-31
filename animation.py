import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def init_scatter(params, ax, points):
    """Function for intializing a scatter artist.

    Sets the sizes and colors of the points that the scatter artist will
    be handling (points for the agents or the visualization dots).

    Parameters
    ----------
        params: dict
            Dictionary which contains parameters for the sizes and colors.
        ax: matplotlib.axes.Axes
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
    scatter = ax.scatter(points[:,0], points[:,1], s=sizes, lw=0.5,
        facecolors=facecolors,  edgecolors=edgecolors)

    return scatter

def init_animation(params, points, dots=None):
    """Function for intializing the figure and artists needed for the animation.

    Sets the figure according to given parameters and intializes the artists
    that will be drawing to it.

    Parameters
    ----------
        params: dict
            Dictionary which contains parameters for the figure and the artists.
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
    ax = fig.add_axes([params['x_min'], params['y_min'], params['x_max'],
        params['y_max']], frameon=params['frameon'])
    ax.set_xlim(params['ax_x_min'], params['ax_x_max'])
    ax.set_ylim(params['ax_y_min'], params['ax_y_max'])

    if params['remove_thicks']:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.grid(params['use_grid'])
    scatter = init_scatter(params, ax, points)

    if dots is not None:
        env_scatter_params = {
            'pointsize': 0.025,
            'edgecolors': [1, 0, 0, 1],
            'facecolors': [1, 0, 0, 1]
            }
        env_scatter = init_scatter(env_scatter_params, ax, dots)

        return fig, env_scatter, scatter

    else:
        return fig, scatter


class Animation(object):
    """Class for the animation object.

    This class is used for showing the agent's movents. If an environment is
    used its vector field can be also visualized in the animation.

    Parameters
    ----------
        plot_params: dict
            Parameters for intializing the animation figure and the artists
            used for the visualizations.
        task_list: list
            List of the agents' movent tasks.
        model:
            Interaction model for the agents and (optionally) the environment.

    """
    def __init__(self, plot_params, task_list, model):
        self.plot_params = plot_params
        self.model = model
        positions = model.positions
        self.task_list = task_list
        self.last = len(task_list) - 1
        self.task_index = 0
        self.use_visuals = False

        if self.model.env and self.model.env.use_visuals:

            self.use_visuals = True
            dots = self.model.env.dots
            fig, env_scatter, scatter = init_animation(
                plot_params, positions, dots)
            self.fig = fig
            self.env_scatter = env_scatter
            self.scatter = scatter

        else:
            fig, scatter = init_animation(plot_params, positions)
            self.fig = fig
            self.scatter = scatter

    def __repr__(self):
        args = self.plot_params, self.task_list, self.model
        return 'Animation({}, {}, {})'.format(*args)

    def update(self, i):
        """Method for updating the animation frames.

        Executes one time unit update of the current movement task and possibly
        sets a new task for the next frame. Also plots the agents postions and
        the vector field visualization.

        Parameters
        ----------
            i: int
                The frame number.

        Returns
        -------
            if self.use_visuals None:
                (scatter,): (
                    matplotlib.axes.collections.PathCollection,
                    )
                Tuple with only the scatter artist for the agents.
            else:
                (env_scatter, scatter): (
                    matplotlib.axes.collections.PathCollection,
                    matplotlib.axes.collections.PathCollection
                    )
                Tuple with the scatter artists for both the agents and the dots.

        """
        if self.task_index > self.last:
            points = self.model.positions

        else:
            task = self.task_list[self.task_index]

            if task['type'] == 'reshape':
                self.model.reshape_formation(*task['args'])

            elif task['type'] == 'shift':
                self.model.shift_formation(*task['args'])

            elif task['type'] == 'turn':
                self.model.turn_formation(*task['args'])

            else:
                raise NotImplementedError

            if self.model.task_params['task_ready']:
                self.task_index +=1

            points = self.model.positions

        self.scatter.set_offsets(points)

        if self.use_visuals:
            dots = self.model.env.visualize()
            self.env_scatter.set_offsets(dots)

            return self.env_scatter, self.scatter

        else:
            return self.scatter,

    def run(self):
        """Runs the animation."""
        animation = FuncAnimation(self.fig, self.update, blit=True)
        plt.show()
