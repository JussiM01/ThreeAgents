import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def init_scatter(params, ax, points):
    """Fuction for intializing a scatter artist.

    Sets the sizes and colors of the points that the scatter artist will
    be handling (points for the agents or the visualization dots).

    Parameters
    ----------
        params: dict
            Dictionary which contains parameters for the sizes and colors.
        ax: matplotlib.axes.Axes
            Axes object of the figure used for the animation.

    Returns:
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

    Returns:
    --------
        if dots are None:
            fig, scatter: (
                matplotlib.figure.Figure,
                matplotlib.axes.collections.PathCollection
                )
            Tuple with the figure and the scatter artist for the agents.
        else:
            fig, env_scatter, scatter: (
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

    def __init__(self, plot_params, task_list, model):

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


    def _update_plot(self, frame_number, points, velocity_vectors, tasks_done):

        # ADD LATER OTHER UPDATES HERE

        self.scatter.set_offsets(points)

        if self.use_visuals:

            dots = self.model.env.visualize()
            self.env_scatter.set_offsets(dots)

    def update(self, i):

        if self.task_index > self.last:
            positions = self.model.positions
            velocities = np.zeros(self.model.positions.shape, dtype=float)
            all_tasks_done = True

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

            if self.model.task_ready:
                self.task_index +=1

            positions = self.model.positions
            velocities = self.model.velocities
            all_tasks_done = False

        self._update_plot(i, positions, velocities, all_tasks_done)

        if self.use_visuals:
            return self.env_scatter, self.scatter

        else:
            return self.scatter,

    def run(self):

        animation = FuncAnimation(self.fig, self.update, blit=True)
        plt.show()
