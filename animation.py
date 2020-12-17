import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def init_canvas(params, points):

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

    def __init__(self, positions, plot_params, task_list, model, env=None):

        fig, scatter = init_canvas(plot_params, positions)
        self.fig = fig
        self.scatter = scatter
        self.model = model
        self.task_list = task_list
        self.last = len(task_list) - 1
        self.task_index = 0

        if env != None:
            raise NotImplementedError


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

            elif task['type'] == 'random': # for testing only
                self.model.random(*task['args']) # (see RandomAgent below)

            else:
                raise NotImplementedError

            if self.model.task_ready:
                self.task_index +=1

            positions = self.model.positions
            velocities = self.model.velocities
            all_tasks_done = False

        self._update_plot(i, positions, velocities, all_tasks_done)


    def run(self):

        animation = FuncAnimation(self.fig, self.update) #, blit=True) # NOTE: MAKE THINGS WORK WITH "blit=True"
        plt.show()


if __name__ == '__main__': # animation testing

    import argparse


    class RandomAgent(object):

        def __init__(self, positions, time_delta=0.05):
            self.positions = positions
            self.velocities = np.zeros(positions.shape, dtype=float)
            self.num_agents = positions.shape[0]
            self.time_delta = time_delta
            self.task_ready = False

        def random(self, scale_x, scele_y):

            wx = np.random.uniform(-scale_x, scale_x, (self.num_agents, 1))
            wy = np.random.uniform(-scale_y, scale_y, (self.num_agents, 1))
            wiggle = np.concatenate([wx, wy], axis=1)

            self.positions += wiggle*self.time_delta
            self.velocities = wiggle


    parser = argparse.ArgumentParser()

    parser.add_argument('-fh', '--fig_hight', type=int, default=7)
    parser.add_argument('-fw', '--fig_width', type=int, default=7)
    parser.add_argument('-x0', '--x_min', type=int, default=0.1)
    parser.add_argument('-x1', '--x_max', type=int, default=0.8)
    parser.add_argument('-y0', '--y_min', type=int, default=0.1)
    parser.add_argument('-y1', '--y_max', type=int, default=0.8)
    parser.add_argument('-fr', '--frameon', type=bool, default=True)
    parser.add_argument('-ax0', '--ax_x_min', type=float, default=-1)
    parser.add_argument('-ax1', '--ax_x_max', type=float, default=1)
    parser.add_argument('-ay0', '--ax_y_min', type=float, default=-1)
    parser.add_argument('-ay1', '--ax_y_max', type=float, default=1)
    parser.add_argument('-u', '--use_grid', type=bool, default=False)
    parser.add_argument('-r', '--remove_thicks', type=bool, default=False)
    parser.add_argument('-p', '--pointsize', type=float, default=50)
    parser.add_argument('-f', '--facecolors', type=str, default='black')
    parser.add_argument('-e', '--edgecolors', type=str, default='black')
    parser.add_argument('-i', '--interval', type=int, default=10)

    args = parser.parse_args()

    points_x = np.random.uniform(args.ax_x_min/10, args.ax_x_max/10, (3, 1))
    points_y = np.random.uniform(args.ax_y_min/10, args.ax_y_max/10, (3, 1))
    points = np.concatenate([points_x, points_y], axis=1)

    random_agent = RandomAgent(points)


    plot_parameters = {
        'fig_hight': args.fig_hight,
        'fig_width': args.fig_width,
        'x_min': args.x_min,
        'x_max': args.x_max,
        'y_min': args.y_min,
        'y_max': args.y_max,
        'frameon': args.frameon,
        'ax_x_min': args.ax_x_min,
        'ax_x_max': args.ax_x_max,
        'ax_y_min': args.ax_y_min,
        'ax_y_max': args.ax_y_max,
        'use_grid': args.use_grid,
        'remove_thicks': args.remove_thicks,
        'pointsize': args.pointsize,
        'facecolors': args.facecolors,
        'edgecolors': args.edgecolors,
        'interval': args.interval
        }

    scale_x = (args.x_max - args.x_min)*0.5
    scale_y = (args.y_max - args.y_min)*0.5
    tasks = [{'type': 'random', 'args': (scale_x, scale_y)}]

    animation = Animation(points, plot_parameters, tasks, random_agent)
    animation.run()
