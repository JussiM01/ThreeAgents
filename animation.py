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

    Attributes
    ----------
        plot_params: dict
            Parameters for intializing the animation figure and the artists
            used for the visualizations.
        model:
            Interaction model for the agents and (optionally) the environment.
        task_list: list
            List of the agents' movent tasks.
        last: int
            Index of the last task.
        task_index: int
            Index of the current task.
        use_visuals: bool
            Wheter or not to visualize disturbance and random topography.
        fig: matplotlib.figure.Figure
            Figure object for the animation plotting.
        scatter: matplotlib.axes.collections.PathCollection
            Scatter object for showing the agents positions in the plot.
        dots: numpy.ndarray (dtype: float)
            (Set only if `use_visuals` is True.)
            Positions of the dots use for visualizing the disturbance.
        countors: matplotlib.contour.QuadContourSet
            (Set only if `use_visuals` is True.)
            Countors for visualizing the random topography.
        env_scatter: matplotlib.axes.collections.PathCollection
            (Set only if `use_visuals` is True.)
            Scatter object for showing the disturbance visualization.

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
            fig, countors, env_scatter, scatter = init_animation(
                plot_params, positions, dots)
            self.fig = fig
            self.countors = countors
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
            if self.use_visuals == False:
                (scatter,): (
                    matplotlib.axes.collections.PathCollection,
                    )
                Tuple with only the scatter artist for the agents.
            if self.use_visuals == True:
                (env_scatter, scatter): (
                    matplotlib.axes.collections.PathCollection,
                    matplotlib.axes.collections.PathCollection
                    )
                Tuple with scatter artists for both the agents and the dots.

        """
        if self.task_index > self.last:
            exit(0)

        else:
            task = self.task_list[self.task_index]

            if task['type'] == 'reshape':
                self.model.reshape_formation(*task['args'])

            elif task['type'] == 'shift':
                self.model.shift_formation(*task['args'])

            elif task['type'] == 'turn':
                self.model.turn_formation(*task['args'])

            elif task['type'] == 'start_acceleration':
                self.model.accelerate('start', task['args'])

            elif task['type'] == 'apply_acceleration':
                self.model.accelerate('apply', task['args'])

            else:
                raise NotImplementedError

            if self.model.task_params['task_ready']:
                self.task_index += 1

            points = self.model.positions

        self.scatter.set_offsets(points)

        if self.use_visuals:
            dots = self.model.env.visualize()
            self.env_scatter.set_offsets(dots)

            return self.env_scatter, self.scatter

        return (self.scatter,)

    def run(self):
        """Runs the animation."""
        _ = FuncAnimation(self.fig, self.update, blit=True)
        plt.show()
