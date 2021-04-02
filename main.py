import argparse
import numpy as np
from interactionmodels import CentralControl
from animation import Animation
from environment import Env, StaticUpFlow
from utils import load_config, random_intial_positions


def main(config_dict, parsed_args):
    """Sets up everything and runs the animation.

    This function first intializes animation, interaction model and environment
    (optional) and then runs the animation.

    Parameters
    ----------
        config_dict: dict
            Dictionary containing the parameters for initializing everything.
        parsed_args: argparse.Namespace
            Parsed arguments containing the parameters that are used for
            sampling the agents' initial positions.

    """
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
