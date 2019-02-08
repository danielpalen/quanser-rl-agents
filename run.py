import os
from pprint import pprint
import gym
import quanser_robots

from common.cmd_util import ArgumentParser
from common.environments import get_env_name
from reps.reps import REPS
from reps.ac_reps import ACREPS
from ppo.ppo import PPO


def run_single_experiment(args=None):

    # if no arguments have been passed, then parse them
    if args is None:
        # prepare arguments
        args = ArgumentParser.parse()
        args.summary_path = os.path.join('out', 'summary', args.name)
        args.checkpoint_path = os.path.join('out', 'models', args.name)

        # make sure the necessary directories exist
        os.makedirs(args.summary_path, exist_ok=True)
        os.makedirs(args.checkpoint_path, exist_ok=True)

        args = vars(args)

    pprint(args)
    print()

    # crete gym env
    args['env'] = gym.make(get_env_name(args['env'], sim=not args['robot']))

    # seed gym env if seed is given
    if args['seed'] is not None:
        args['env'].seed(args.seed)
        print(f'# Seeded Env. Seed={args["seed"]}')

    # select, instantiate and train correct algorithm
    model = {'REPS': REPS, 'ACREPS': ACREPS, 'PPO': PPO}[args['algorithm']](**args)
    model.train()
    args['env'].close()


if __name__ == '__main__':
    run_single_experiment()
