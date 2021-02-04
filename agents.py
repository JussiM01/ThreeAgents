import argparse
import numpy as np
import copy


class MultiAgent(object):

    def __init__(self, positions, target_distance, bond_strength, max_speed,
            time_delta, accepted_error, env=None):

        self.positions = copy.deepcopy(positions)
        self.velocities = np.zeros(positions.shape)
        self.targeted_positions = copy.deepcopy(positions)
        self.disturbancies = np.zeros(positions.shape)
        self.target_distance =  target_distance
        self.bond_strength = bond_strength
        self.max_speed = max_speed
        self.num_agents = positions.shape[0]
        self.time_delta = time_delta
        self.accepted_error = accepted_error
        self.formation_type = None
        self.course_target = None
        self.course_direction = None
        self.course_speed = None
        self.rotation_center = None
        self.dist_center_to_points = None
        self.target_direction = None
        self.target_vectors = None
        self.rotation_speed = None
        self.lead_index = None
        self.rotation_sign = None
        self.task_ready = True
        self.env = env


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


        def _straight_move_update(self, uncut_velocities): # NOTE: SOMETHING ELSE NEEDED FOR _turn_step

            pure_velocities = self._cliped(uncut_velocities)
            self.velocities = pure_velocities

            if self.env is None:

                self.positions += self.velocities*self.time_delta
                self.targeted_positions += self.velocities*self.time_delta

            else:

                disturbancies = self.env.evaluate(self.positions)
                self.disturbancies = disturbancies
                disturbed_velocities = pure_velocities + disturbancies

                self.positions += disturbed_velocities*self.time_delta
                self.targeted_positions += pure_velocities*self.time_delta


    def reshape_formation(self, formation_type, speed):

        self.task_ready = False

        if formation_type == 'triangle':
            self.formation_type = formation_type
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

        if self.formation_type == 'triangle':

            for i in range(self.num_agents):
                vectors_to_all = self.positions - self.positions[i,:]
                deviations = np.linalg.norm(
                    vectors_to_all, axis=1) - self.target_distance
                deviations = np.expand_dims(deviations, axis=1)
                velocity = self.bond_strength*speed*np.sum(
                    deviations*self._directions(vectors_to_all), axis=0)
                velocities.append(velocity)

        velocities = np.stack(velocities, axis=0)
        self._straight_move_update(velocities)


    def turn_formation(self, target_point, speed):
        '''Turn the formation around its center of mass until it faces the
        direction of a given target point.'''

        if self.formation_type != 'triangle':
            raise NotImplementedError

        if self.rotation_center is None:
            self.task_ready = False

            center_of_mass = np.mean(self.positions, axis=0)
            to_target = np.array(target_point, dtype=float) - center_of_mass
            direction = self._direction(to_target)
            cliped_speed = speed if speed <= self.max_speed else self.max_speed

            self.dist_center_to_points = self.target_distance/np.sqrt(3)
            self.rotation_center = center_of_mass
            self.target_direction = direction
            self.rotation_speed = cliped_speed
            self.lead_index = self._closest_to(center_of_mass, direction)
            self.rotation_sign = self._rotation_sign(center_of_mass, direction,
                self.lead_index)

            target_vector = self.target_direction*self.dist_center_to_points
            self.target_vectors = np.stack([self._rotate(target_vector, theta)
                for theta in (0, 2*np.pi/3, 4*np.pi/3)])

        lead_position = self.positions[self.lead_index]
        lead_direction = self._direction(lead_position - self.rotation_center)
        direction_diff = np.linalg.norm(lead_direction - self.target_direction)

        if direction_diff < self.accepted_error:

            self.rotation_center = None
            self.target_direction = None
            self.rotation_speed = None
            self.lead_index = None
            self.rotation_sign = None
            self.task_ready = True

        else:

            angle = (self.rotation_speed*self.time_delta
                /self.dist_center_to_points)

            if self._about_to_over_turn(direction_diff, angle):

                vecs = copy.deepcopy(self.positions) - self.rotation_center

                for i in range(3):
                    ind = np.argmin(np.linalg.norm(vecs[i]
                        - self.target_vectors, axis=1))
                    new_point = self.rotation_center + self.target_vectors[ind]

                    self.positions[i] = new_point
                    self.velocities[i] = (new_point - vecs[i])/self.time_delta

            else:
                self._turn_step(angle, speed)


    def _turn_step(self, angle, speed):

        center_to_points = self.positions - self.rotation_center
        new_points = self.rotation_center + self._rotate_all(center_to_points,
            angle*self.rotation_sign)
        diff_vectors = new_points - center_to_points
        diff_directions = self._directions(diff_vectors)

        self.positions = new_points # NOTE: CHANGE TO WORK WITH ENV
        self.velocities = diff_directions*speed # NOTE: CHANGE TO WORK WITH ENV


    def _about_to_over_turn(self, direction_diff, angle):

        lead_point = self.positions[self.lead_index]
        lead_direction = self._direction(lead_point - self.rotation_center)

        planned_direction = self._rotate(lead_direction, angle*self.rotation_sign)
        planned_diff = np.linalg.norm(planned_direction - self.target_direction)

        return planned_diff > direction_diff


    def _closest_to(self, center_point, direction_to_target):

        center_to_points = self.positions - center_point
        directions = self._directions(center_to_points)
        differences = np.linalg.norm(directions - direction_to_target, axis=1)

        return np.argmin(differences)


    def _conjugate_product(self, vector1, vector2):

        vec1_complex = np.array([vector1[0] + 1j*vector1[1]])
        vec2_complex = np.array([vector2[0] + 1j*vector2[1]])

        return vec1_complex*vec2_complex.conj()


    def _rotation_sign(self, center_point, direction_to_target, lead_index):

        center_to_point = self.positions[lead_index,:] - center_point
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
        (and make course corrections if there are any disturbancies).''' # NOTE: ADD COURSE CORRECTION LATER

        center_of_mass = np.mean(self.positions, axis=0)

        if self.course_target is None:

            self.task_ready = False
            self.course_target = np.array(target_point, dtype=float)
            self.course_direction = self._direction(
                center_of_mass - self.course_target)
            self.course_speed = speed

        cm_to_target = self.course_target - center_of_mass
        dist_cm_to_target = np.linalg.norm(cm_to_target)

        if dist_cm_to_target < self.accepted_error:

            self.course_target = None
            self.course_direction = None
            self.course_speed = None
            self.task_ready = True

        else:

            if self._about_to_over_shoots(center_of_mass):
                adjusted_direction = self._direction(cm_to_target)
                adjusted_speed = dist_cm_to_target/self.time_delta

            else:
                adjusted_direction = self.course_direction
                adjusted_speed = self.course_speed

            self._shift_step(adjusted_direction, adjusted_speed)


    def _about_to_over_shoots(self, current_cm):

        dist_to_target = np.linalg.norm(self.course_target - current_cm)
        velocity = self.course_direction*self.course_speed

        planned_move = velocity*self.time_delta
        planned_next = current_cm + planned_move
        planned_dist = np.linalg.norm(self.course_target - planned_next)

        return planned_dist > dist_to_target


    def _shift_step(self, direction, speed):
        '''Move all agents to given direction with the given speed without
        changing the angle of the formation.'''
        velocities = np.tile(direction*speed, [self.num_agents, 1])
        self._straight_move_update(velocities)
