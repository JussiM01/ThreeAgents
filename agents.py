import argparse
import numpy as np



class MultiAgent(object):

    def __init__(self, positions, target_distance, bond_strength, max_speed,
            time_delta, accepted_error):

        self.positions = positions
        self.formation_velocities = np.zeros(positions.shape)
        self.rotation_velocities = np.zeros(positions.shape)
        self.velocities = np.zeros(positions.shape)
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
        self.target_direction = None
        self.rotation_speed = None
        self.lead_index = None
        self.rotation_sign = None
        self.task_ready = True


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
        self.velocities = self._cliped(velocities)
        self.positions += self.velocities*self.time_delta


    def turn_formation(self, target_point, speed):
        '''Turn the formation around its center of mass until it faces the
        direction of a given target point.'''

        # if self.formation_type != 'triangle':
        #     raise NotImplementedError
        #
        # if self.rotation_center is None:
        #     self.task_ready = False
        #
        #     center_of_mass = np.mean(self.positions, axis=0)
        #     to_target = target_point - center_of_mass
        #     direction = self._direction(to_target)
        #
        #     self.rotation_center = center_of_mass
        #     self.target_direction = direction
        #     self.rotation_speed = speed
        #     self.lead_index = self._closest_to(center_of_mass, direction)
        #     self.rotation_sign = self._rotation_sign(center_of_mass, direction,
        #         self.lead_index)
        #
        # lead_postion = self.postions[self.lead_index]
        # lead_direction = self._(lead_direction - self.rotation_center)
        # direction_diff = np.linalg.norm(lead_direction - self.target_direction)
        #
        # if direction_diff < self.accepted_error:
        #
        #     self.rotation_center = None
        #     self.target_direction = None
        #     self.rotation_speed = None
        #     self.lead_index = None
        #     self.rotation_sign = None
        #     self.task_ready = True
        #
        # ###############################
        #
        # else:
        #
        #     if self._about_to_over_turn(....):
        #         adjusted_direction = self._direction(cm_to_target)
        #         adjusted_speed = dist_cm_to_target/self.time_delta
        #
        #     else:
        #         adjusted_direction = self.course_direction
        #         adjusted_speed = self.course_speed
        #
        #     self._turn_step(adjusted_direction, adjusted_speed)
        #
        # #########################

        raise NotImplementedError


    # def _turn_step(self): # NOTE: NEEDS MORE ARGUMENTS ?
    #
    #     raise NotImplementedError
    #
    #
    # def _about_to_over_turn(self): # NOTE: NEEDS MORE ARGUMENTS ?
    #
    #     raise NotImplementedError


    def _closest_to(self, center_point, direction_to_target):

        center_to_points = self.postions - center_point
        directions = self._directions(center_to_points)
        differences = np.linalg.norm(directions - direction_to_target, axis=1)

        return np.argmin(differences)


    def _rotation_sign(self, center_point, direction_to_target, lead_index):

        center_to_point = self.postions[lead_index,:] - center_point
        direction = self._direction(center_to_point)

        direction_complex = np.array([direction[0] + direction[1]*j])
        to_target_complex = np.array([direction_to_target[0] +
            direction_to_target[1]*j])

        conj_to_target = to_target_complex.conj()
        product = direction_complex*conj_to_target

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
        self.velocities = self._cliped(velocities)
        self.positions += self.velocities*self.time_delta



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--random_seed', type=int, default=0)
    parser.add_argument('-n', '--num_agents', type=int, default=3)
    parser.add_argument('-d', '--distance', type=float, default=10.0)
    parser.add_argument('-b', '--bond_value', type=float, default=10.0)
    parser.add_argument('-to', '--top_speed', type=float, default=1.0)
    parser.add_argument('-td', '--time_delta', type=float, default=0.01)
    parser.add_argument('-e', '--episode_lenght', type=int, default=1000)
    parser.add_argument('-a', '--accepted_error', type=float, default=1e-3)

    args = parser.parse_args()

    initial_positions = np.random.uniform(0, 10, (args.num_agents, 2))

    model = MultiAgent(
        initial_positions,
        args.distance,
        args.bond_value,
        args.top_speed,
        args.time_delta,
        args.accepted_error
        )

    # print('\n### moving all points to same direction ###\n')
    #
    # for i in range(args.episode_lenght):
    #     model.move_all(np.array([1.0, 1.0]), 1.0)
    #     print('\n\nFRAME NUMBER: ', i)
    #     print('\nPOSITIONS:\n', model.positions)
    #     print('\nvelocities:\n', model.velocities)

    print('\n### adjusting the fromation ###\n')

    for i in range(args.episode_lenght):

        if i == 0:
            model.reshape_formation('triangle', args.top_speed)

        elif (i !=0) and (not model.task_ready):
            model.reshape_formation('triangle', args.top_speed)

        else:
            print('TASK READY')
            break

        print('\n\nFRAME NUMBER: ', i)
        print('\nPOSITIONS:\n', model.positions)
        print('\nvelocities:\n', model.velocities)
