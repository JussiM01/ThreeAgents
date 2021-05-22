import numpy as np

from copy import deepcopy
from utils import normalize, normalize_all, rotate


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
        self.disturbance = np.zeros(position.shape)
        self.max_speed = max_speed
        self.time_delta = time_delta
        self.accepted_error = accepted_error

    def _move(self, uncut_velocity, disturbance, use_correction=True):
        """Method for moving the agent position based on its velocity.

        Moves the agent according to the velocity and the disturbance from the
        environment (if not equal to None) and the course corrections. The
        targeted position is updated similarly but without the disturbancies.

        Parameters
        ----------
            uncut_velocities: numpy.ndarray (dtype: float)
                Array of shape (2, ) containing the velocity (without
                corrections and disturbancies) of each agent.
            disturbance: numpy.ndarray (dtype: float) or None
                Array of shape (2, ) containing the disturbance from the
                environment (or None if environment is equal to None).
            use_correction: bool (optional, default: True)
                Boolean telling wether to use course correction or not.

        """
        pure_velocity = self._clip(uncut_velocity)
        self.targeted_velocity = pure_velocity

        if disturbance is None:
            self.velocity = pure_velocity
            self.position += self.velocity*self.time_delta
            self.targeted_position += self.velocity*self.time_delta

        else:
            old_disturbance = self.disturbance
            self.disturbance = disturbance

            if use_correction:
                corrected_velocity = self._course_correction(pure_velocity)

            else:
                corrected_velocity = pure_velocity

            self.velocity = corrected_velocity
            disturbed_velocity = corrected_velocity + old_disturbance
            self.position += disturbed_velocity*self.time_delta
            self.targeted_position += pure_velocity*self.time_delta

    def _clip(self, velocity):
        speed = np.linalg.norm(velocity)
        if speed <= self.max_speed:
            return velocity
        return normalize(velocity)*self.max_speed


class LeadAgent(BaseAgent):
    """Class for representing the leading agent.

    This class is used for modelling the leading agent which is reponsible for
    the execution of the movement task while the other agents are following it.
    The lead agent is also the only one responsible for making  the course
    corrections if there are disturbancies from the enivronment present.

    Parameters
    ----------
        position: numpy.ndarray (dtype: float)
            Array of shape (2, ) representing the agent's position.
        max_speed: float
            Maximum speed of the agent.
        time_delta: float
            Lenght of the unit time increment.
        accepted_error: float
            Tasks are done when distance to target is less than this.
        correction_const: list (types: [float, float])
            Constants for velocity and position based course corrections.

    """

    def __init__(self, position, max_speed, time_delta, accepted_error,
                 correction_const=None):
        super().__init__(position, max_speed, time_delta, accepted_error)

        if correction_const is None:
            self.correction_const = [1.0, 1.0]
        else:
            self.correction_const = correction_const

        self.action_params = {'task_ready': True}

    def __repr__(self):
        args = (self.position, self.max_speed, self.time_delta,
                self.accepted_error, self.correction_const)
        repr = 'LeadAgent({}, {}, {}, {}, {})'
        return repr.format(*args)

    def shift(self, target_point, speed, disturbance):
        """Shifts the lead agent towards the givent target point.

        Moves the agents towards the target point with the given speed and
        makes the applies course corrections if there are any disturbancies.
        The task is considered finished when the targeted_position is within
        distance of `self.accepted_error` away from target.

        Parameters
        ----------
            target_point: numpy.ndarray (dtype: float)
                Array of shape (2, ), coordinates of the target point.
            speed: flot
                speed of the shifting.
            disturbance: numpy.ndarray (dtype: float) or None (default: None)
                Array of shape (2, ), the disturbance vector (None if the
                environment is equal to None).

        """
        if 'course_target' not in self.action_params:
            self.action_params['task_ready'] = False
            self.action_params['course_target'] = np.array(
                target_point, dtype=float)
            self.action_params['course_direction'] = normalize(
                self.action_params['course_target'] - deepcopy(self.position))
            self.action_params['course_speed'] = speed

        to_target = (self.action_params['course_target']
                     - deepcopy(self.targeted_position))
        dist_to_target = np.linalg.norm(to_target)

        if self._about_to_over_shoots():
            adjusted_direction = normalize(to_target)
            adjusted_speed = dist_to_target/self.time_delta

        else:
            adjusted_direction = self.action_params['course_direction']
            adjusted_speed = self.action_params['course_speed']

        self._move(adjusted_direction*adjusted_speed, disturbance)

    def _about_to_over_shoots(self):
        """Checks if the agent is about to pass the target."""
        current_diff = (self.targeted_position
            - self.action_params['course_target'])
        velocity = (self.action_params['course_direction']
                    * self.action_params['course_speed'])
        planned_move = velocity*self.time_delta
        planned_next = self.targeted_position + planned_move
        planned_diff = planned_next - self.action_params['course_target']

        return current_diff.dot(planned_diff) < 0

    def start_accelerate(self, strength, direction, disturbance=None):
        """Initiates acceleration towards given direction.

        This method is ment for starting the agents accelration when it is at
        rest. If there is disturbance from the environment present its effect
        is also modelled in the movement.

        Parameters
        ----------
            strength: float
                Strenght of the acceleration.
            direction: numpy.ndarray (dtype: float)
                Array of shape (2, ), the direction of the acceleration.
            disturbance: numpy.ndarray (dtype: float) or None (default: None)
                Array of shape (2, ), the disturbance vector (None if the
                environment is equal to None).

        """
        acceleration = strength*direction
        velocity = acceleration*self.time_delta

        self._move(velocity, disturbance)

    def apply_accelerate(self, tangential, normal, disturbance=None):
        """Adds tangential and normal acceleration to the velocity.

        This method is ment to be used when the agent is already moving and
        one wants to add tangential and/or normal acceration to its movement.
        The normal direction is current velocity direction rotated 90 degrees
        to clockwise. If there is disturbance from the environment it will be
        added to the movement.

        Parameters
        ----------
            tangential: float
                Strength of the tangential acceleration.
            normal: float
                Strength of the normal acceleration.
            disturbance:  numpy.ndarray (dtype: float) or None (default: None)
                Array of shape (2, ), the disturbance vector (None if the
                environment is equal to None).

        """
        tangential_direction = normalize(self.velocity)
        normal_direction = rotate(tangential_direction, np.pi/2)
        tangential_acceleration = tangential*tangential_direction
        normal_acceleration = normal*normal_direction
        velocity_diff = (tangential_acceleration
                         + normal_acceleration)*self.time_delta
        velocity = self.velocity + velocity_diff

        self._move(velocity, disturbance)

    def _course_correction(self, velocity):
        """Applies course correction to the velocity."""
        velocity_diff = self.targeted_velocity - velocity
        position_diff = self.targeted_position - self.position
        adjusted_velocity = (
            velocity + self.correction_const[0]*velocity_diff
            + self.correction_const[1]*position_diff/self.time_delta)

        return self._clip(adjusted_velocity)


class FollowerAgent(BaseAgent):
    """Class for representing the follower agents.

    This class is used for modelling those two agents that are following the
    lead agent. Their primary task is to always try to stay within target
    distance away from both the lead agent and each other.

    Parameters
    ----------
        position: numpy.ndarray (dtype: float)
            Array of shape (2, ) representing the agent's position.
        target_distance: float
            Target distance between the agents in the formation.
        bond_strength: float
            Strenght constant for the formation reshaping.
        max_speed: float
            Maximum speed of the agent.
        time_delta: float
            Lenght of the unit time increment.
        accepted_error: float
            Tasks are done when distance to target is less than this.

     """

    def __init__(self, position, target_distance, bond_strength, max_speed,
                 time_delta, accepted_error):
        super().__init__(position, max_speed, time_delta, accepted_error)
        self.target_distance = target_distance
        self.bond_strength = bond_strength

    def __repr__(self):
        args = (self.position, self.target_distance, self.bond_strength,
                self.max_speed, self.time_delta, self.accepted_error)
        repr = 'FollowerAgent({}, {}, {}, {}, {}, {}, {}, {}, {})'
        return repr.format(*args)

    def keep_distance(self, other_positions, speed, disturbance):
        """Keeps the agents distancies close to the target distance."""
        vectors_to_others = other_positions - self.position
        deviations = (np.linalg.norm(vectors_to_others, axis=1)
                      - self.target_distance)
        deviations = np.expand_dims(deviations, axis=1)
        velocity = self.bond_strength*speed*np.sum(
            deviations*normalize_all(vectors_to_others), axis=0)

        self._move(velocity, disturbance, False)
