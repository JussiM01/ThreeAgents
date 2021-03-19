import argparse
import numpy as np
import copy


class BaseModel(object):
    """Base model for the interactionmodels.

    This class is used as a container for basic attributes and shared private
    methods. The actual functionality is implemented within its subclasses.
    """

    def __init__(self, positions, max_speed, time_delta, accepted_error,
            env=None):
        """Inits the BaseModel with agents' initial positions, their maximum
        speed, unit time lenght, allowed error and optionally the environment.
        """
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
        """Moves the agents according to their velocities, which include
        disturbancies from the environment (if not equal to None) and
        course corrections.
            The targeted positions are updated similarly but without the
        disturbancies.

        Parameters
        ----------
            uncut_velocites: numpy.ndarray (dtype: float)
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

    def _direction(self, vector):
        norm = np.linalg.norm(vector)
        return vector if norm == 0 else vector/norm

    def _directions(self, vectors):
        return np.apply_along_axis(lambda x: self._direction(x), 1, vectors)

    def _clip(self, velocity):
        speed = np.linalg.norm(velocity)
        if speed <= self.max_speed:
            return velocity
        else:
            return self._direction(velocity)*self.max_speed

    def _cliped(self, velocities):
        speeds = np.linalg.norm(velocities, axis=1)
        if np.max(speeds) <= self.max_speed:
            return velocities
        else:
            return np.apply_along_axis(lambda x: self._clip(x), 1, velocities)


class CentralControl(BaseModel):

    def __init__(self, positions, target_distance, bond_strength, max_speed,
            time_delta, accepted_error, env=None, correction_const=[1.0, 1.0]):

        super().__init__(positions, max_speed, time_delta, accepted_error, env)

        self.target_distance =  target_distance
        self.bond_strength = bond_strength
        self.correction_const = correction_const
        self.task_params = {}
        self.task_ready = True

    def _course_correction(self, velocities):

        velocities_diff = self.targeted_velocities - velocities
        positions_diff = self.targeted_positions - self.positions
        adjusted_velocities = (velocities +
            self.correction_const[0]*velocities_diff +
            self.correction_const[1]*positions_diff/self.time_delta)

        return self._cliped(adjusted_velocities)

    def reshape_formation(self, formation_type, speed):

        self.task_ready = False

        if formation_type == 'triangle':
            self.task_params['formation_type'] = formation_type
            dist_01 = np.linalg.norm(self.positions[0] - self.positions[1])
            dist_02 = np.linalg.norm(self.positions[0] - self.positions[2])
            dist_12 = np.linalg.norm(self.positions[1] - self.positions[2])

            error = max([abs(d - self.target_distance)
                for d in [dist_01, dist_02, dist_12]])

            if error < self.accepted_error:
                self.task_ready = True

            else:
                self._reshape_step(speed)

        else:
            raise NotImplementedError


    def _reshape_step(self, speed):

        velocities = []

        if self.task_params['formation_type'] == 'triangle':

            for i in range(self.num_agents):
                vectors_to_all = self.positions - self.positions[i,:]
                deviations = np.linalg.norm(
                    vectors_to_all, axis=1) - self.target_distance
                deviations = np.expand_dims(deviations, axis=1)
                velocity = self.bond_strength*speed*np.sum(
                    deviations*self._directions(vectors_to_all), axis=0)
                velocities.append(velocity)

        velocities = np.stack(velocities, axis=0)
        self._move(velocities)


    def turn_formation(self, target_point, speed):
        '''Turn the formation around its center of mass until it faces the
        direction of a given target point.'''

        if self.task_params['formation_type'] != 'triangle':
            raise NotImplementedError

        if 'rotation_center' not in self.task_params:
            self.task_ready = False

            center_of_mass = np.mean(self.targeted_positions, axis=0)
            to_target = np.array(target_point, dtype=float) - center_of_mass
            direction = self._direction(to_target)
            cliped_speed = speed if speed <= self.max_speed else self.max_speed

            self.task_params['dist_center_to_points'] = (self.target_distance
                /np.sqrt(3))
            self.task_params['rotation_center'] = center_of_mass
            self.task_params['target_direction'] = direction
            self.task_params['rotation_speed'] = cliped_speed
            self.task_params['lead_index'] = self._closest_to(
                center_of_mass, direction)
            self.task_params['rotation_sign'] = self._rotation_sign(
                center_of_mass, direction, self.task_params['lead_index'])

            target_vector = (self.task_params['target_direction']
                *self.task_params['dist_center_to_points'])
            self.task_params['target_vectors'] = np.stack(
                [self._rotate(target_vector, theta)
                for theta in (0, 2*np.pi/3, 4*np.pi/3)])

        lead_position = self.targeted_positions[self.task_params['lead_index']]
        lead_direction = self._direction(lead_position
            - self.task_params['rotation_center'])
        direction_diff = np.linalg.norm(lead_direction
            - self.task_params['target_direction'])

        if direction_diff < self.accepted_error:

            self.task_params = {}
            self.task_ready = True

        else:

            angle = (self.task_params['rotation_speed']*self.time_delta
                /self.task_params['dist_center_to_points'])

            if self._about_to_over_turn(direction_diff, angle):

                vecs = (copy.deepcopy(self.targeted_positions)
                    - self.task_params['rotation_center'])

                for i in range(3):
                    ind = np.argmin(np.linalg.norm(vecs[i]
                        - self.task_params['target_vectors'], axis=1))
                    new_point = (self.task_params['rotation_center']
                        + self.task_params['target_vectors'][ind])

                    if self.env is None:

                        self.targeted_positions[i] = new_point
                        self.positions[i] = new_point
                        self.velocities[i] = (
                            new_point - vecs[i])/self.time_delta

                    else:

                        self.targeted_positions[i] = new_point
                        self.velocities[i] = (
                            new_point - vecs[i])/self.time_delta
                        disturbance = self.env.evaluate(
                            np.expand_dims(new_point, axis=0))
                        self.positions[i] = (new_point +
                            disturbance[0]*self.time_delta)
                        self.disturbancies[i] = disturbance

            else:
                self._turn_step(angle, speed)


    def _turn_step(self, angle, speed):

        center_to_points = (self.targeted_positions
            - self.task_params['rotation_center'])
        new_points = self.task_params['rotation_center'] + self._rotate_all(
            center_to_points, angle*self.task_params['rotation_sign'])
        diff_vectors = new_points - center_to_points
        diff_directions = self._directions(diff_vectors)
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

        lead_point = self.targeted_positions[self.task_params['lead_index']]
        lead_direction = self._direction(lead_point
            - self.task_params['rotation_center'])

        planned_direction = self._rotate(
            lead_direction, angle*self.task_params['rotation_sign'])
        planned_diff = np.linalg.norm(planned_direction
            - self.task_params['target_direction'])

        return planned_diff > direction_diff


    def _closest_to(self, center_point, direction_to_target):

        center_to_points = self.targeted_positions - center_point
        directions = self._directions(center_to_points)
        differences = np.linalg.norm(directions - direction_to_target, axis=1)

        return np.argmin(differences)


    def _conjugate_product(self, vector1, vector2):

        vec1_complex = np.array([vector1[0] + 1j*vector1[1]])
        vec2_complex = np.array([vector2[0] + 1j*vector2[1]])

        return vec1_complex*vec2_complex.conj()


    def _rotation_sign(self, center_point, direction_to_target, lead_index):

        center_to_point = self.targeted_positions[lead_index,:] - center_point
        direction = self._direction(center_to_point)
        product = self._conjugate_product(direction, direction_to_target)

        return -1*np.sign(product.imag)[0]


    def _rotate(self, vector, angle):

        rotation_matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

        return rotation_matrix.dot(vector)


    def _rotate_all(self, points, angle):

        return np.apply_along_axis(lambda x: self._rotate(x, angle), 1, points)


    def shift_formation(self, target_point, speed):
        '''Move the formation towards the target point with the given speed.
        (and make course corrections if there are any disturbancies).'''

        center_of_mass = np.mean(self.targeted_positions, axis=0)

        if 'course_target' not in self.task_params:

            self.task_ready = False
            self.task_params['course_target'] = np.array(
                target_point, dtype=float)
            self.task_params['course_direction'] = self._direction(
                self.task_params['course_target'] - center_of_mass)
            self.task_params['course_speed'] = speed

        cm_to_target = self.task_params['course_target'] - center_of_mass
        dist_cm_to_target = np.linalg.norm(cm_to_target)

        if dist_cm_to_target < self.accepted_error:

            self.task_params = {}
            self.task_ready = True

        else:

            if self._about_to_over_shoots(center_of_mass):
                adjusted_direction = self._direction(cm_to_target)
                adjusted_speed = dist_cm_to_target/self.time_delta

            else:
                adjusted_direction = self.task_params['course_direction']
                adjusted_speed = self.task_params['course_speed']

            self._shift_step(adjusted_direction, adjusted_speed)


    def _about_to_over_shoots(self, current_cm):

        current_diff = current_cm - self.task_params['course_target']
        velocity = (self.task_params['course_direction']
            *self.task_params['course_speed'])

        planned_move = velocity*self.time_delta
        planned_next = current_cm + planned_move
        planned_diff = planned_next - self.task_params['course_target']

        return current_diff.dot(planned_diff) < 0


    def _shift_step(self, direction, speed):
        '''Move all agents to given direction with the given speed without
        changing the angle of the formation.'''
        velocities = np.tile(direction*speed, [self.num_agents, 1])
        self._move(velocities)
