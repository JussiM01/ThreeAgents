import numpy as np


class BaseAgent:
    """Base class for lead and follower agents.

    This class is used as a container for basic attributes and shared private
    methods. The actual functionality is implemented within its subclasses.

    """
    def __init__(self, position, time_delta):
        self.position = position
        self.time_delta = time_delta
        self.velocity = np.zeros(position.shape)

    def _move(self, velocity):
        self.position += self.time_delta*velocity
        self.velocity = velocity
