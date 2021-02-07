import argparse
import json
import os
import numpy as np
from agents import MultiAgent
from animation import Animation
from environment import Env, StaticUpFlow # NOTE: ADD MORE LATER



def load_config(filename):
    config_file = os.path.join('config_files',  filename)
    with open(config_file, 'r') as f:
        config = json.load(f)

    return config


parser = argparse.ArgumentParser()

# Random seed and relative range of the agents starting points.
parser.add_argument('-r', '--random_seed', type=int, default=0)
parser.add_argument('-x0', '--x_min', type=float, default=0.1)
parser.add_argument('-x1', '--x_max', type=float, default=0.2)
parser.add_argument('-y0', '--y_min', type=float, default=0.4)
parser.add_argument('-y1', '--y_max', type=float, default=0.5)

# Config-file for the agents, animation and tasks (movements of the agents).
parser.add_argument('-f', '--conf_file', type=str, default='basic_config.json')

args = parser.parse_args()


np.random.seed(args.random_seed)

config = load_config(args.conf_file)

anim_init = config['animation']
model_init = config['model']
env_init = config['env']
tasks = config['tasks']


x_axis_len = anim_init['ax_x_max'] - anim_init['ax_x_min']
y_axis_len = anim_init['ax_y_max'] - anim_init['ax_y_min']

# Set the starting ranges for the agents positions.
start_min_x = anim_init['ax_x_min'] + x_axis_len*args.x_min
start_max_x = anim_init['ax_x_min'] + x_axis_len*args.x_max
start_min_y = anim_init['ax_y_min'] + y_axis_len*args.y_min
start_max_y = anim_init['ax_y_min'] + y_axis_len*args.y_max

# Choose randomly intial positions for the agents.
points_x = np.random.uniform(start_min_x, start_max_x, (3, 1))
points_y = np.random.uniform(start_min_y, start_max_y, (3, 1))
initial_positions = np.concatenate([points_x, points_y], axis=1)


model_init['positions'] = initial_positions

if env_init['vectorfield'] is not None:

    vectorfield = StaticUpFlow(**env_init['params'])
    model_init['env'] = Env(vectorfield, model_init['time_delta'])

agents = MultiAgent(**model_init)

animation = Animation(initial_positions, anim_init, tasks, agents)
animation.run()
