import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def init_animation(params, points):

    fig = plt.figure(figsize=(params['fig_width'], params['fig_hight']))
    ax = fig.add_axes([params['x_min'], params['y_min'], params['x_max'],
        params['y_max']], frameon=params['frameon'])

    ax.set_xlim(params['ax_x_min'], params['ax_x_max'])
    ax.set_ylim(params['ax_y_min'], params['ax_y_max'])

    if params['remove_thicks']:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.grid(params['use_grid'])

    size = np.array((params['pointsize']), dtype=float)
    sizes = np.repeat((size), points.shape[0], axis=0)

    if params['facecolors'] == 'black':
        facecolor = np.array([[0, 0, 0, 1]], dtype=float)

    else:
        raise NotImplementedError

    facecolors = np.repeat(facecolor, points.shape[0], axis=0)

    if params['edgecolors'] == 'black':
        edgecolor = np.array([[0, 0, 0, 1]], dtype=float)

    else:
        raise NotImplementedError

    edgecolors = np.repeat(edgecolor, points.shape[0], axis=0)

    scatter = ax.scatter(points[:,0], points[:,1], s=sizes, lw=0.5,
        facecolors=facecolors,  edgecolors=edgecolors)

    # NOTE: ADD LATER GOAL POINT(S) PLOTTING & STATS WRITING !

    return fig, scatter



class Animation(object):

    def __init__(self, positions, plot_params, task_list, model):

        fig, scatter = init_animation(plot_params, positions)
        self.fig = fig
        self.scatter = scatter
        self.model = model
        self.task_list = task_list
        self.last = len(task_list) - 1
        self.task_index = 0


    def _update_plot(self, frame_number, points, velocity_vectors, tasks_done):

        # ADD LATER OTHER UPDATES HERE

        self.scatter.set_offsets(points)


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

        return self.scatter,


    def run(self):

        animation = FuncAnimation(self.fig, self.update, blit=True)
        plt.show()
