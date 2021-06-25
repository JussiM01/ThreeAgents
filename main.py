"""A simple multi-agent simulation with three agents moving in a formation.

Configuration for the simulation along with the movement tasks for the agents
are given in a configuration file. For more details see the project's README.

"""
import argparse

import numpy as np

from animation import Animation
from environment import Env, StaticUpFlow
from interactionmodels import CentralControl, OneLead
from utils import load_config, random_intial_positions


def main(parsed_args):
    """Sets up everything and runs the simulation.

    This function first intializes animation, interaction model and environment
    (optional) and then runs the simulation.

    Parameters
    ----------
        parsed_args: argparse.Namespace
            Parsed arguments containing the parameters that are used for
            sampling the agents' initial positions.

    """
    config_dict = load_config(parsed_args.conf_file)
    interaction = config_dict['interaction']
    anim_init = config_dict['animation']
    model_init = config_dict['model']
    env_init = config_dict['env']
    tasks = config_dict['tasks']

    initial_positions = random_intial_positions(anim_init, parsed_args)
    model_init['positions'] = initial_positions

    if env_init['vectorfield'] is not None:
        vectorfield = StaticUpFlow(**env_init['params'])
        visuals_init = env_init['visuals']
        time_delta = model_init['time_delta']
        model_init['env'] = Env(vectorfield, time_delta, visuals_init)

    if interaction == 'central_control':
        interactionmodel = CentralControl(**model_init)

    if interaction == 'one_lead':
        interactionmodel = OneLead(**model_init)

    if parsed_args.use_thicks:
        anim_init['remove_thicks'] = False

    if parsed_args.use_grid:
        anim_init['remove_thicks'] = False
        anim_init['use_grid'] = True

    animation = Animation(anim_init, tasks, interactionmodel)
    animation.run()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '-f', '--conf_file', type=str, default='visuals_cen.json')
    parser.add_argument('-x0', '--x_min', type=float, default=0.1)
    parser.add_argument('-x1', '--x_max', type=float, default=0.2)
    parser.add_argument('-y0', '--y_min', type=float, default=0.4)
    parser.add_argument('-y1', '--y_max', type=float, default=0.5)
    parser.add_argument('-t', '--use_thicks', action='store_true')
    parser.add_argument('-g', '--use_grid', action='store_true')
    parser.add_argument('-r', '--random_seed', type=int)
    args = parser.parse_args()

    # set random seed
    np.random.seed(args.random_seed)

    main(args)
