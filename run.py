import os
import yaml
from pprint import pprint
import gym
import quanser_robots

from common.cmd_util import ArgumentParser
from common.environments import get_env_name
from reps.reps import REPS
from reps.acreps import ACREPS
from ppo.ppo import PPO


def run_single_experiment(args=None):

    # if no arguments have been passed, then load defaults and parse them
    if args is None:
        # prepare arguments
        args = ArgumentParser.parse()
        # Load defaults
        defaults_path = os.path.join('hyperparameters', args.algo.lower(), f"{args.env}.yaml")
        print(defaults_path)

        with open(defaults_path) as f:
            params = yaml.load(f)['params']
            print('DEFAULTS')
            pprint(params)

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
    args['robot'] = args['robot'] if 'robot' in args else False
    args['env'] = gym.make(get_env_name(args['env'], sim=not args['robot']))

    # seed gym env if seed is given
    args['seed'] = args['seed'] if 'seed' in args else None
    if args['seed'] is not None:
        args['env'].seed(args["seed"])
        print(f'# Seeded Env. Seed={args["seed"]}')

    # select, instantiate and train correct algorithm
    model = {'REPS': REPS, 'ACREPS': ACREPS, 'PPO': PPO}[args['algo'].upper()](**args)
    print(model)
    if 'eval' in args and args['eval']:
        model.evaluate(100)
    else:
        model.train()
    args['env'].close()


if __name__ == '__main__':
    run_single_experiment()
