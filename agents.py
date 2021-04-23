import numpy as np
from copy import deepcopy
from utils import normalize


class BaseAgent:
    """Base class for lead and follower agents.

    This class is used as a container for basic attributes and shared private
    methods. The actual functionality is implemented within its subclasses.

    """
    def __init__(self, position, max_speed, time_delta, accepted_error):
        self.position = deepcopy(position)
        self.velocity = np.zeros(position.shape)
        self.targeted_position = deepcopy(position)
        self.targeted_velocity = np.zeros(position.shape)
        self.max_speed = max_speed
        self.time_delta = time_delta
        self.accepted_error = accepted_error

    def _move(self, uncut_velocity, disturbance):

        pure_velocity = self._cliped(uncut_velocity)
        self.targeted_velocity = pure_velocity

        if disturbance is None:
            self.velocity = pure_velocity
            self.position += self.velocity*self.time_delta
            self.targeted_position += self.velocity*self.time_delta

        else:
            raise NotImplementedError

    def _clip(self, velocity):
        speed = np.linalg.norm(velocity)
        if speed <= self.max_speed:
            return velocity
        return normalize(velocity)*self.max_speed


class LeadAgent(BaseAgent):
    """Class for representing the leading agent."""

    def __init__(self, position, max_speed, time_delta, accepted_error,
                 correction_const=None, task_params=None):
        super().__init__(position, max_speed, time_delta, accepted_error)
        self.targeted_positions = copy.deepcopy(positions)
        self.targeted_velocities = np.zeros(positions.shape)

        if correction_const is None:
            self.correction_const = [1.0, 1.0]
        else:
            self.correction_const = correction_const

        if task_params is None:
            self.task_params = {'task_ready': True}
        else:
            self.task_params = task_params

    def shift(self, target_point, speed, disturbance):

        if disturbance is not None:
            raise NotImplementedError

        if 'course_target' not in self.task_params:
            self.task_params['task_ready'] = False
            self.task_params['course_target'] = np.array(
                target_point, dtype=float)
            self.task_params['course_direction'] = normalize(
                self.task_params['course_target'] - deepcopy(self.position))
            self.task_params['course_speed'] = speed

        to_target = self.task_params['course_target']
            - deepcopy(self.position)
        dist_to_target = np.linalg.norm(to_target)

        if to_target < self.accepted_error:
            self.task_params = {'task_ready': True}

        else:
            if self._about_to_over_shoots():
                adjusted_direction = normalize(to_target)
                adjusted_speed = dist_to_target/self.time_delta

            else:
                adjusted_direction = self.task_params['course_direction']
                adjusted_speed = self.task_params['course_speed']

            self._move(adjusted_direction*adjusted_speed, disturbance)

    def _about_to_over_shoots(self):
        """Checks if the agent is about to pass the target."""
        current_diff = current_cm - self.task_params['course_target']
        velocity = (self.task_params['course_direction']
                    * self.task_params['course_speed'])
        planned_move = velocity*self.time_delta
        planned_next = self.position + planned_move
        planned_diff = planned_next - self.task_params['course_target']

        return current_diff.dot(planned_diff) < 0

    def follow_path(self, path):

        raise NotImplementedError

    def _course_correction(self, velocity):

        velocity_diff = self.targeted_velocity - velocity
        position_diff = self.targeted_position - self.position
        adjusted_velocity = (
            velocity + self.correction_const[0]*velocity_diff
            + self.correction_const[1]*position_diff/self.time_delta)

        return self._clip(adjusted_velocity)


class FollowerAgent(BaseAgent):
    """Class for representing the follower agents."""

    def __init__(self, position, target_distance, bond_strength, max_speed,
                 time_delta, accepted_error):
        super().__init__(position, max_speed, time_delta, accepted_error)
        self.target_distance = target_distance
        self.bond_strength = bond_strength

    def keep_distance(self, other_positions, speed, disturbance):

        vectors_to_others = other_positions - self.position
        deviations = np.linalg.norm(
            vectors_to_others, axis=1) - self.target_distance
        deviations = np.expand_dims(deviations, axis=1)
        velocity = self.bond_strength*speed*np.sum(
            deviations*normalize_all(vectors_to_others), axis=0)

        self._move(velocity, disturbance)
