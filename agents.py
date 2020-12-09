import argparse
import numpy as np



class MultiAgent(object):

    def __init__(self, positions, target_distance, bond_strength, max_velocity,
            time_delta, accepted_error):

        self.positions = positions
        self.formation_speeds = np.zeros(positions.shape)
        self.rotation_speeds = np.zeros(positions.shape)
        self.speeds = np.zeros(positions.shape)
        self.target_distance =  target_distance
        self.bond_strength = bond_strength
        self.max_velocity = max_velocity
        self.num_agents = positions.shape[0]
        self.time_delta = time_delta
        self.accepted_error = accepted_error
        self.course_target = None
        self.course_direction = None
        self.course_velocity = None
        self.task_ready = True


    def _direction(self, vector):

        norm = np.linalg.norm(vector)

        return vector if norm == 0 else vector/norm


    def _directions(self, vectors):

        return np.apply_along_axis(lambda x: self._direction(x), 1, vectors)


    def _clip(self, speed):

        velocity = np.linalg.norm(speed)

        if velocity <= self.max_velocity:
            return speed

        else:
            return self._direction(speed)*self.max_velocity


    def _cliped(self, speeds):

        velocities = np.linalg.norm(speeds, axis=1)

        if np.max(velocities) <= self.max_velocity:
            return speeds

        else:
            return np.apply_along_axis(lambda x: self._clip(x), 1, speeds)


    def reshape_formation(self, formation_type, velocity):

        self.task_ready = False

        if formation_type == 'triangle':
            dist_01 = np.linalg.norm(self.positions[0] - self.positions[1])
            dist_02 = np.linalg.norm(self.positions[0] - self.positions[2])
            dist_12 = np.linalg.norm(self.positions[1] - self.positions[2])

            error = abs(max([dist_01, dist_02, dist_12]) - self.target_distance)

            if error < self.accepted_error:
                self.task_ready = True

            else:
                self._reshape_step(formation_type, velocity)

        else:
            raise NotImplementedError


    def _reshape_step(self, formation_type, velocity):

        speeds = []

        if formation_type == 'triangle':

            for i in range(self.num_agents):
                vectors_to_all = self.positions - self.positions[i,:]
                deviations = np.linalg.norm(
                    vectors_to_all, axis=1) - self.target_distance
                deviations = np.expand_dims(deviations, axis=1)
                speed = self.bond_strength*velocity*np.sum(
                    deviations*self._directions(vectors_to_all), axis=0)
                speeds.append(speed)

        speeds = np.stack(speeds, axis=0)
        self.speeds = self._cliped(speeds)
        self.positions += self.speeds*self.time_delta


    def turn_formation(self, direction, velocity):
        '''Turn the formation around its center of mass until it faces the given
        direction.'''

        raise NotImplementedError


    def shift_formation(self, target_point, velocity):
        '''Move the formation towards the target point with the given velocity.
        (and make course corrections if there are any disturbancies).''' # NOTE: ADD COURSE CORRECTION LATER

        center_of_mass = np.mean(selff.positions, axis=0)

        if self.course_target == None:

            self.task_ready = False
            self.course_target = target_point
            self.course_direction = self._direction(
                center_of_mass - self.course_target)
            self.course_velocity = velocity

        dist_cm_to_target = np.linalg.norm(center_of_mass - self.course_target)

        if dist_cm_to_target < self.accepted_error:

            self.tasks_done = True

        else:

            self._shift_step(self.course_direction, self.course_velocity)


    def _shift_step(self, direction, velocity):
        '''Move all agents to given direction with the given velocity without
        changing the angle of the formation.'''
        
        speeds = np.tile(direction*velocity, [self.num_agents, 1])
        self.speeds = self._cliped(speeds)
        self.positions += self.speeds*self.time_delta



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--random_seed', type=int, default=0)
    parser.add_argument('-n', '--num_agents', type=int, default=3)
    parser.add_argument('-d', '--distance', type=float, default=10.0)
    parser.add_argument('-b', '--bond_value', type=float, default=10.0)
    parser.add_argument('-to', '--top_velocity', type=float, default=1.0)
    parser.add_argument('-td', '--time_delta', type=float, default=0.01)
    parser.add_argument('-e', '--episode_lenght', type=int, default=1000)
    parser.add_argument('-a', '--accepted_error', type=float, default=1e-3)

    args = parser.parse_args()

    initial_positions = np.random.uniform(0, 10, (args.num_agents, 2))

    model = MultiAgent(
        initial_positions,
        args.distance,
        args.bond_value,
        args.top_velocity,
        args.time_delta,
        args.accepted_error
        )

    # print('\n### moving all points to same direction ###\n')
    #
    # for i in range(args.episode_lenght):
    #     model.move_all(np.array([1.0, 1.0]), 1.0)
    #     print('\n\nFRAME NUMBER: ', i)
    #     print('\nPOSITIONS:\n', model.positions)
    #     print('\nSPEEDS:\n', model.speeds)

    print('\n### adjusting the fromation ###\n')

    for i in range(args.episode_lenght):

        if i == 0:
            model.reshape_formation('triangle', args.top_velocity)

        elif (i !=0) and (not model.task_ready):
            model.reshape_formation('triangle', args.top_velocity)

        else:
            print('TASK READY')
            break

        print('\n\nFRAME NUMBER: ', i)
        print('\nPOSITIONS:\n', model.positions)
        print('\nSPEEDS:\n', model.speeds)
