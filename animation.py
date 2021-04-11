import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils import init_animation


class Animation:
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

        return self.scatter,

    def run(self):
        """Runs the animation."""
        _ = FuncAnimation(self.fig, self.update, blit=True)
        plt.show()
