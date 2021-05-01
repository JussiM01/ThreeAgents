import copy
import numpy as np
from agents import FollowerAgent, LeadAgent
from utils import (conjugate_product, normalize, normalize_all, rotate,
                   rotate_all)


class BaseModel:
    """Base model for the interactionmodels.

    This class is used as a container for basic attributes and shared private
    methods. The actual functionality is implemented within its subclasses.

    """

    def __init__(self, positions, max_speed, time_delta, accepted_error,
                 env=None):
        self.positions = copy.deepcopy(positions)
        self.velocities = np.zeros(positions.shape)
        self.targeted_positions = copy.deepcopy(positions)
        self.targeted_velocities = np.zeros(positions.shape)
        self.disturbancies = np.zeros(positions.shape)
        self.max_speed = max_speed
        self.num_agents = positions.shape[0]
        self.time_delta = time_delta
        self.accepted_error = accepted_error
        self.env = env

    def _move(self, uncut_velocities):
        """
        Method for moving the agents positions based on their velocites.

        Moves the agents according to their velocities, which include
        disturbancies from the environment (if not equal to None) and
        course corrections.
            The targeted positions are updated similarly but without the
        disturbancies.

        Parameters
        ----------
            uncut_velocities: numpy.ndarray (dtype: float)
                Array of shape (3, 2) containing the velocities (without
                corrections and disturbancies) of each agent.

        """
        pure_velocities = self._cliped(uncut_velocities)
        self.targeted_velocities = pure_velocities

        if self.env is None:
            self.velocities = pure_velocities
            self.positions += self.velocities*self.time_delta
            self.targeted_positions += self.velocities*self.time_delta

        else:
            old_disturbancies = self.disturbancies
            disturbancies = self.env.evaluate(self.positions)
            self.disturbancies = disturbancies
            velocities = pure_velocities + old_disturbancies
            corrected_velocities = self._course_correction(velocities)
            disturbed_velocities = corrected_velocities + disturbancies
            self.positions += disturbed_velocities*self.time_delta
            self.targeted_positions += pure_velocities*self.time_delta
            self.velocities = corrected_velocities

    def _course_correction(self, velocities):
        raise NotImplementedError

    def _clip(self, velocity):
        speed = np.linalg.norm(velocity)
        if speed <= self.max_speed:
            return velocity
        return normalize(velocity)*self.max_speed

    def _cliped(self, velocities):
        speeds = np.linalg.norm(velocities, axis=1)
        if np.max(speeds) <= self.max_speed:
            return velocities
        return np.apply_along_axis(lambda x: self._clip(x), 1, velocities)


class CentralControl(BaseModel):
    """Class representing centrally controlled agents.

    This class models agents that execute tasks from central controller,
    optionally under disturbances from environment. In the later situation
    the controller applies also course corrections.

    Parameters
    ----------
        positions: numpy.ndarray (dtype: float)
            Array of shape (3, 2) representing the agents positions.
        target_distance: float
            Target distance between the agents in the triangle formation.
        bond_strength: float
            Strenght constant for the triangle formation reshaping.
        max_speed: float
            Maximum allowed speed for all agents.
        time_delta: float
            Lenght of the unit time increment.
        accepted_error: float
            Tasks are done when distance to target is less than this.
        env: environment.Env
            Object representing the environment.
        correction_const: list (types: [float, float])
            Constants for velocity and position based course corrections.

    """

    def __init__(self, positions, target_distance, bond_strength, max_speed,
                 time_delta, accepted_error, env=None, correction_const=None,
                 task_params=None):
        super().__init__(positions, max_speed, time_delta, accepted_error, env)
        self.target_distance = target_distance
        self.bond_strength = bond_strength

        if correction_const is None:
            self.correction_const = [1.0, 1.0]
        else:
            self.correction_const = correction_const

        if task_params is None:
            self.task_params = {'task_ready': True}
        else:
            self.task_params = task_params

    def __repr__(self):
        args = (self.positions, self.target_distance, self.bond_strength,
                self.max_speed, self.time_delta, self.accepted_error, self.env,
                self.correction_const, self.task_params)
        repr = 'CentralControl({}, {}, {}, {}, {}, {}, {}, {}, {})'
        return repr.format(*args)

    def _course_correction(self, velocities):
        """Method for updatting veclocities to include course corrections.

        Adds velocity components for velocity and position based course
        corrections to the given velocities. These are calculated based
        on previous disturbancies and attempt to change the velocities
        and positions towards their targeted versions.

        Parameters
        ----------
            velocities: numpy.ndarray (dtype: float)
                Array of shape (3, 2) containing the velocities of each agent.

        Returns
        -------

            adjusted_velocities: numpy.ndarray (dtype: float)
                Array of shape (3, 2) containing the course corrected velocites
                of each agent.

        """
        velocities_diff = self.targeted_velocities - velocities
        positions_diff = self.targeted_positions - self.positions
        adjusted_velocities = (
            velocities + self.correction_const[0]*velocities_diff
            + self.correction_const[1]*positions_diff/self.time_delta)

        return self._cliped(adjusted_velocities)

    def reshape_formation(self, formation_type, speed):
        """Method for reshaping the agents' formation to given shape

        Moves the agent towards positions fo the given formation type. This
        task is accepted as done when the postions are close enough to wanted
        positions (ie. the distance is less than `self.accepted_error`).

        Parameters
        ----------
            formation_type: str
                Name of the formation type (currently only the `triangle`
                formation type is supported).
            speed: float
                Speed parameter for the reshaping (see `_reshape_step` for
                details).

        """
        self.task_params['task_ready'] = False

        if formation_type == 'triangle':
            self.task_params['formation_type'] = formation_type
            dist_01 = np.linalg.norm(self.positions[0] - self.positions[1])
            dist_02 = np.linalg.norm(self.positions[0] - self.positions[2])
            dist_12 = np.linalg.norm(self.positions[1] - self.positions[2])

            error = max([abs(d - self.target_distance)
                        for d in [dist_01, dist_02, dist_12]])

            if error < self.accepted_error:
                self.task_params = {
                    'task_ready': True, 'formation_type': formation_type}

            else:
                self._reshape_step(speed)

        else:
            raise NotImplementedError

    def _reshape_step(self, speed):
        """Method for single time step formation reshaping.

        This method is run when reshape_formation is called, unless the
        acceptance criteria is met. It moves the agent towards desired form.
        Currently only triangle shape is supported. In it the agents are
        in an uniform triangle within `self.target_distance` away from
        each other (or reasonably close to it). The agents movement speeds
        are based on their deviation from the desired distance, speed and the
        attribute `self.bond_strength`. The movement velocities are products
        of these (see the code for exact details).

        Parameters
        ----------
            speed: float
                Speed parameter for the agent reshape movements.

        """
        velocities = []

        if self.task_params['formation_type'] == 'triangle':
            for i in range(self.num_agents):
                vectors_to_all = self.positions - self.positions[i, :]
                deviations = np.linalg.norm(
                    vectors_to_all, axis=1) - self.target_distance
                deviations = np.expand_dims(deviations, axis=1)
                velocity = self.bond_strength*speed*np.sum(
                    deviations*normalize_all(vectors_to_all), axis=0)
                velocities.append(velocity)

        velocities = np.stack(velocities, axis=0)
        self._move(velocities)

    def turn_formation(self, target_point, speed):
        """Method for turning the formation around its center point.

        Turns the formation around its center until it faces the direction of
        given target point (upto maximum error of `self.accepted_error`).

        Parameters
        ----------
            target_point: list (types [float, float])
                List containing the coordinates of the target point.
            speed: flot
                speed of the rotation.

        """
        if self.task_params['formation_type'] != 'triangle':
            raise NotImplementedError

        if 'rotation_center' not in self.task_params:
            self.task_params['task_ready'] = False
            center_of_mass = np.mean(self.targeted_positions, axis=0)
            to_target = np.array(target_point, dtype=float) - center_of_mass
            direction = normalize(to_target)
            cliped_speed = speed if speed <= self.max_speed else self.max_speed
            self.task_params['dist_center_to_points'] = (self.target_distance
                                                         / np.sqrt(3))
            self.task_params['rotation_center'] = center_of_mass
            self.task_params['target_direction'] = direction
            self.task_params['rotation_speed'] = cliped_speed
            self.task_params['lead_index'] = self._closest_to(
                center_of_mass, direction)
            self.task_params['rotation_sign'] = self._rotation_sign(
                center_of_mass, direction, self.task_params['lead_index'])
            target_vector = (self.task_params['target_direction']
                             * self.task_params['dist_center_to_points'])
            self.task_params['target_vectors'] = np.stack(
                [rotate(target_vector, theta)
                 for theta in (0, 2*np.pi/3, 4*np.pi/3)])

        lead_position = self.targeted_positions[self.task_params['lead_index']]
        lead_direction = normalize(
            lead_position - self.task_params['rotation_center'])
        direction_diff = np.linalg.norm(
            lead_direction - self.task_params['target_direction'])

        if direction_diff < self.accepted_error:
            formation_type = self.task_params['formation_type']
            self.task_params = {
                'task_ready': True, 'formation_type': formation_type}

        else:
            angle = (self.task_params['rotation_speed'] * self.time_delta
                     / self.task_params['dist_center_to_points'])

            if self._about_to_over_turn(direction_diff, angle):
                vecs = (copy.deepcopy(self.targeted_positions)
                        - self.task_params['rotation_center'])
                for i in range(3):
                    ind = np.argmin(np.linalg.norm(
                        vecs[i] - self.task_params['target_vectors'], axis=1))
                    new_point = (self.task_params['rotation_center']
                                 + self.task_params['target_vectors'][ind])

                    if self.env is None:
                        self.targeted_positions[i] = new_point
                        self.positions[i] = new_point
                        self.velocities[i] = (
                            new_point - vecs[i]) / self.time_delta

                    else:
                        self.targeted_positions[i] = new_point
                        self.velocities[i] = (
                            new_point - vecs[i])/self.time_delta
                        disturbance = self.env.evaluate(
                            np.expand_dims(new_point, axis=0))
                        self.positions[i] = (new_point + disturbance[0]
                                             * self.time_delta)
                        self.disturbancies[i] = disturbance

            else:
                self._turn_step(angle, speed)

    def _turn_step(self, angle, speed):
        """Method for single time step formation turning movement.

        This method is run when turn_formation method is called. It tunrs the
        formation around its center by given angle. The angle and speed are
        calculated in the turn_formation method and depend on each other but
        are both passed as args for convienience.

        Parameters
        ----------
            angle: float
                Angle of the turn.
            speed: float
                Speed that is needed for this movement.

        """
        center_to_points = (self.targeted_positions
                            - self.task_params['rotation_center'])
        new_points = self.task_params['rotation_center'] + rotate_all(
            center_to_points, angle*self.task_params['rotation_sign'])
        diff_vectors = new_points - center_to_points
        diff_directions = normalize_all(diff_vectors)
        self.targeted_positions = new_points
        self.targeted_velocities = diff_directions*speed

        if self.env is None:
            self.positions = copy.deepcopy(new_points)
            self.velocities = diff_directions*speed

        else:
            old_disturbancies = self.disturbancies
            disturbancies = self.env.evaluate(self.positions)
            self.disturbancies = disturbancies
            velocities = diff_directions*speed + old_disturbancies
            corrected_velocities = self._course_correction(velocities)
            self.positions += (corrected_velocities +
                               disturbancies)*self.time_delta
            self.velocities = corrected_velocities + disturbancies

    def _about_to_over_turn(self, direction_diff, angle):
        """Checks if the formation is about to turn more than intended."""
        lead_point = self.targeted_positions[self.task_params['lead_index']]
        lead_direction = normalize(
            lead_point - self.task_params['rotation_center'])
        planned_direction = rotate(
            lead_direction, angle*self.task_params['rotation_sign'])
        planned_diff = np.linalg.norm(planned_direction
                                      - self.task_params['target_direction'])

        return planned_diff > direction_diff

    def _closest_to(self, center_point, direction_to_target):
        """Finds index of the agent that is closest to target direction."""
        center_to_points = self.targeted_positions - center_point
        directions = normalize_all(center_to_points)
        differences = np.linalg.norm(directions - direction_to_target, axis=1)

        return np.argmin(differences)

    def _rotation_sign(self, center_point, direction_to_target, lead_index):
        """Calculates rotation sign (clockwise or counter clockwise)."""
        center_to_point = self.targeted_positions[lead_index, :] - center_point
        direction = normalize(center_to_point)
        product = conjugate_product(direction, direction_to_target)

        return -1*np.sign(product.imag)[0]

    def shift_formation(self, target_point, speed):
        """Method for shifting the formation towards a given target point.

        Moves the formation towards the target point with the given speed
        and make course corrections if there are disturbancies. The task
        is finished when center point of the formation is within distance
        of `self.accepted_error` away from target.

        Parameters
        ----------
            target_point: list (types [float, float])
                List containing the coordinates of the target point.
            speed: flot
                speed of the shifting.

        """

        center_of_mass = np.mean(self.targeted_positions, axis=0)

        if 'course_target' not in self.task_params:
            self.task_params['task_ready'] = False
            self.task_params['course_target'] = np.array(
                target_point, dtype=float)
            self.task_params['course_direction'] = normalize(
                self.task_params['course_target'] - center_of_mass)
            self.task_params['course_speed'] = speed

        cm_to_target = self.task_params['course_target'] - center_of_mass
        dist_cm_to_target = np.linalg.norm(cm_to_target)

        if dist_cm_to_target < self.accepted_error:
            formation_type = self.task_params['formation_type']
            self.task_params = {
                'task_ready': True, 'formation_type': formation_type}

        else:

            if self._about_to_over_shoots(center_of_mass):
                adjusted_direction = normalize(cm_to_target)
                adjusted_speed = dist_cm_to_target/self.time_delta

            else:
                adjusted_direction = self.task_params['course_direction']
                adjusted_speed = self.task_params['course_speed']

            self._shift_step(adjusted_direction, adjusted_speed)

    def _about_to_over_shoots(self, current_cm):
        """Checks if the formation is about to pass the target."""
        current_diff = current_cm - self.task_params['course_target']
        velocity = (self.task_params['course_direction']
                    * self.task_params['course_speed'])
        planned_move = velocity*self.time_delta
        planned_next = current_cm + planned_move
        planned_diff = planned_next - self.task_params['course_target']

        return current_diff.dot(planned_diff) < 0

    def _shift_step(self, direction, speed):
        """Moves all agents to given direction with the given speed."""
        velocities = np.tile(direction*speed, [self.num_agents, 1])
        self._move(velocities)


class OneLead(BaseModel):
    """Class for interaction model with one lead agent and two followers."""

    def __init__(self, positions, target_distance, bond_strength, max_speed,
                 time_delta, accepted_error, env=None, correction_const=None,
                 task_params=None):
        super().__init__(positions, max_speed, time_delta, accepted_error, env)
        self.lead_agent = LeadAgent(
            positions[0, :], max_speed, time_delta, accepted_error,
            correction_const, task_params)
        self.follower1 = FollowerAgent(
            positions[1, :], target_distance, bond_strength, max_speed,
            time_delta, accepted_error)
        self.follower2 = FollowerAgent(
            positions[2, :], target_distance, bond_strength, max_speed,
            time_delta, accepted_error)
        self.target_distance = target_distance
        self.task_params = self.lead_agent.task_params
        self.env = env

    def reshape_formation(self, speed):

        dist_01 = np.linalg.norm(self.positions[0] - self.positions[1])
        dist_02 = np.linalg.norm(self.positions[0] - self.positions[2])
        dist_12 = np.linalg.norm(self.positions[1] - self.positions[2])

        error = max([abs(d - self.target_distance)
                    for d in [dist_01, dist_02, dist_12]])

        if error < self.accepted_error:
            self.task_params['task_ready'] = True

        else:
            self.task_params['task_ready'] = False
            self._reshape_moves(speed)

    def _reshape_moves(self, speed):

        if self.env is None:
            self._follow_lead(speed)
            self.lead_agent.shift(
                self.task_params['course_target'],
                self.task_params['course_speed'], None)

        else:
            disturbancies = self.env.evaluate(self.positions)
            self._follow_lead(speed, disturbancies[[1, 2], :])
            self.lead_agent._move(
                np.array([0., 0.], dtype=float), disturbancies[0, :])

        self._update_state()

    def shift_formation(self, target_point, speed):

        if 'course_target' not in self.task_params:
            self.task_params['task_ready'] = False
            self.task_params['course_target'] = np.array(
                target_point, dtype=float)
            self.task_params['course_direction'] = normalize(
                self.task_params['course_target'] - self.positions[0, :])
            self.task_params['course_speed'] = speed

        error = np.linalg.norm(
            self.positions[0, :] - self.task_params['course_target'])

        if error < self.accepted_error:
            self.task_params['task_ready'] = True

        else:
            self._shift_moves(speed)

    def _shift_moves(self, speed):

        if self.env is None:
            self._follow_lead(speed)
            self.lead_agent.shift(
                self.task_params['course_target'],
                self.task_params['course_speed'], None)

        else:
            disturbancies = self.env.evaluate(self.positions)
            self._follow_lead(speed, disturbancies[[1, 2], :])
            self.lead_agent.shift(
                self.task_params['course_target'],
                self.task_params['course_speed'], disturbancies[0, :])

        self._update_state()

    def accelerate(self, acceleration_type, parameters):

        if acceleration_type == 'start':
            strength, direction, duration, follow_speed = parameters

        if acceleration_type == 'apply':
            tangential, normal, duration, follow_speed = parameters

        if 'duration' not in self.task_params:
            self.task_params = {'task_ready': False, 'duration': duration}

        if self.task_params['duration'] == 0:
            self.task_params = {'task_ready': True}

        else:
            if acceleration_type == 'start':
                self._start_step(strength, direction, follow_speed)

            if acceleration_type == 'apply':
                self._apply_step(tangential, normal, follow_speed)

            self.task_params['duration'] -= 1

    def _start_step(self, strength, direction, follow_speed):

        direction = np.array(direction, dtype=float)

        if self.env is None:
            self._follow_lead(follow_speed)
            self.lead_agent.start_accelerate(strength, direction)

        else:
            disturbancies = self.env.evaluate(self.positions)
            self._follow_lead(follow_speed, disturbancies[[1, 2], :])
            self.lead_agent.start_accelerate(
                strength, direction, disturbancies[0, :])

    def _apply_step(self, tangential, normal, follow_speed):

        if np.linalg.norm(self.lead_agent.velocity) == 0:
            msg = "Velocity is zero. Use `start_acceleration`."
            raise ValueError(msg)

        if self.env is None:
            self._follow_lead(follow_speed)
            self.lead_agent.apply_accelerate(tangential, normal)

        else:
            disturbancies = self.env.evaluate(self.positions)
            self._follow_lead(follow_speed, disturbancies[[1, 2], :])
            self.lead_agent.apply_accelerate(
                tangential, normal, disturbancies[0, :])

    def _follow_lead(self, speed, followers_disturbancies=None):

        other_positions1 = copy.deepcopy(self.positions[[0, 2], :])
        other_positions2 = copy.deepcopy(self.positions[[0, 1], :])

        if followers_disturbancies is None:
            self.follower1.keep_distance(other_positions1, speed, None)
            self.follower2.keep_distance(other_positions2, speed, None)

        else:
            self.follower1.keep_distance(
                other_positions1, speed, followers_disturbancies[0, :])
            self.follower2.keep_distance(
                other_positions2, speed, followers_disturbancies[1, :])

        self._update_state()

    def _update_state(self):

         pos0 = copy.deepcopy(self.lead_agent.position)
         pos1 = copy.deepcopy(self.follower1.position)
         pos2 = copy.deepcopy(self.follower2.position)
         self.positions = np.stack([pos0, pos1, pos2], axis=0)

         vel0 = copy.deepcopy(self.lead_agent.velocity)
         vel1 = copy.deepcopy(self.follower1.velocity)
         vel2 = copy.deepcopy(self.follower2.velocity)
         self.velocities = np.stack([vel0, vel1, vel2], axis=0)
