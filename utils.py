import json
import os
import numpy as np


#---- Helpers for main.py ------------------------------------------------------

def load_config(filename):
    """Loads the configuration file."""
    config_file = os.path.join('config_files',  filename)
    with open(config_file, 'r') as f:
        config = json.load(f)

    return config

def random_intial_positions(anim_dict, parsed_args):
    """Intializes the agents' positions randomly according to a given range."""
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

#---- Helpers for interactionmodels.py -----------------------------------------

def conjugate_product(vector1, vector2):
    """Multiplies as complex numbers vector1 and conjugate of vector2."""
    vec1_complex = np.array([vector1[0] + 1j*vector1[1]])
    vec2_complex = np.array([vector2[0] + 1j*vector2[1]])

    return vec1_complex*vec2_complex.conj()

def rotate(vector, angle):
    rotation_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    return rotation_matrix.dot(vector)

def rotate_all(points, angle):
    return np.apply_along_axis(lambda x: rotate(x, angle), 1, points)
