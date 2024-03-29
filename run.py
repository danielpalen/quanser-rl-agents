import os
import yaml
from pprint import pprint
import gym
import quanser_robots

from common.cmd_util import ArgumentParser
from common.environments import get_env_name
from agents.reps import REPS
from agents.acreps import ACREPS
from agents.ppo import PPO


def run_single_experiment(args=None):
    """
    This method is the main entry point for running an experiment. It trys to set everything up,
    instantiate a model and train and/or evaluate a model.

    If no args dictionary is supplied the arguemnt parser will try to parse all
    necessary arguments from the command line and combine them with the default
    arguments that are located in ./hyperparameters/[algorithm]/[environment].yaml.

    :param args: dictionary of arguments that is used to set up everything including the model.
    """

    # if no arguments have been passed, then load defaults and parse them
    if args is None:
        # prepare arguments
        args = ArgumentParser.parse()
        # Load defaults
        defaults_path = os.path.join('hyperparameters', args.algo.lower(), f"{args.env}.yaml")
        print(defaults_path)

        params = {}
        if os.path.exists(defaults_path):
            with open(defaults_path) as f:
                params = yaml.load(f, Loader=yaml.FullLoader)['params']
                #print('DEFAULTS')
                #pprint(params)

        if args.experiment:
            args.summary_path = os.path.join('out', 'experiments', '_'.join(args.name.split('_')[:-1]), 'summary', args.name)
            args.checkpoint_path = os.path.join('out', 'experiments', '_'.join(args.name.split('_')[:-1]), 'models', args.name)
        else:
            args.summary_path = os.path.join('out', 'summary', args.name)
            args.checkpoint_path = os.path.join('out', 'models', args.name)

        # make sure the necessary directories exist
        os.makedirs(args.summary_path, exist_ok=True)
        os.makedirs(args.checkpoint_path, exist_ok=True)

        args = vars(args)
        for k, v in args.items():
            if v is not None:
                params[k] = v
        params['seed'] = args['seed']  # has to be set extra, because seed=None is a valid option
        args = params

    print('ARGUMENTS')
    pprint(args)
    print()


    args['eval'] = args['eval'] if 'eval' in args else False
    args['resume'] = args['resume'] if 'resume' in args else False
    args['robot'] = args['robot'] if 'robot' in args else False
    args['seed'] = args['seed'] if 'seed' in args else None

    if not args['resume'] and not args['eval']:
        # save arguments
        params_path = os.path.join(args['checkpoint_path'], 'params.yaml')
        with open(params_path, 'w') as outfile:
            yaml.dump({'params' : args}, outfile, default_flow_style=None)
            print('saved args! location:', os.path.join(args['checkpoint_path'], 'params.yaml'))
    #else:
        # TODO: should we overwrite command line args with previous values?
        # load arguments from previous run

    # crete gym env
    args['env'] = gym.make(get_env_name(args['env'], sim=not args['robot']))

    # seed gym env if seed is given
    if args['seed'] is not None:
        args['env'].seed(args["seed"])
        print(f'# Seeded Env. Seed={args["seed"]}')

    # select, instantiate and train correct algorithm
    model = {'REPS': REPS, 'ACREPS': ACREPS, 'PPO': PPO}[args['algo'].upper()](**args)
    print(model)
    if 'eval' in args and args['eval']:
        model.evaluate(args['n_eval_traj'], print_reward=True)
    else:
        model.train()
    args['env'].close()


if __name__ == '__main__':
    run_single_experiment()
