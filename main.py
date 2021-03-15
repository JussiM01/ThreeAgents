import argparse
import json
import os
import numpy as np
from interactionmodels import CentralControl
from animation import Animation
from environment import Env, StaticUpFlow



def load_config(filename):
    config_file = os.path.join('config_files',  filename)
    with open(config_file, 'r') as f:
        config = json.load(f)

    return config

def random_intial_positions(anim_dict, parsed_args):
    x_axis_len = anim_dict['ax_x_max'] - anim_dict['ax_x_min']
    y_axis_len = anim_dict['ax_y_max'] - anim_dict['ax_y_min']

    # set the starting ranges for the agents intial positions
    start_min_x = anim_dict['ax_x_min'] + x_axis_len*parsed_args.x_min
    start_max_x = anim_dict['ax_x_min'] + x_axis_len*parsed_args.x_max
    start_min_y = anim_dict['ax_y_min'] + y_axis_len*parsed_args.y_min
    start_max_y = anim_dict['ax_y_min'] + y_axis_len*parsed_args.y_max

    # choose the agents intial positions randomly from these ranges
    points_x = np.random.uniform(start_min_x, start_max_x, (3, 1))
    points_y = np.random.uniform(start_min_y, start_max_y, (3, 1))
    initial_positions = np.concatenate([points_x, points_y], axis=1)

    return initial_positions

def main(config_dict, parsed_args):
    anim_init = config_dict['animation']
    model_init = config_dict['model']
    env_init = config_dict['env']
    tasks = config_dict['tasks']

    initial_positions = random_intial_positions(anim_init, args)
    model_init['positions'] = initial_positions

    if env_init['vectorfield'] is not None:
        vectorfield = StaticUpFlow(**env_init['params'])
        visuals_init = env_init['visuals']
        time_delta = model_init['time_delta']
        model_init['env'] = Env(vectorfield, time_delta, visuals_init)

    interactionmodel = CentralControl(**model_init)

    animation = Animation(anim_init, tasks, interactionmodel)
    animation.run()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--conf_file', type=str, default='visuals.json')
    parser.add_argument('-r', '--random_seed', type=int, default=0)
    parser.add_argument('-x0', '--x_min', type=float, default=0.1)
    parser.add_argument('-x1', '--x_max', type=float, default=0.2)
    parser.add_argument('-y0', '--y_min', type=float, default=0.4)
    parser.add_argument('-y1', '--y_max', type=float, default=0.5)
    args = parser.parse_args()

    # set random seed
    np.random.seed(args.random_seed)

    # load config
    config = load_config(args.conf_file)

    main(config, args)
