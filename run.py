from pprint import pprint
import gym
import quanser_robots

from common.cmd_util import ArgumentParser
from common.environments import get_env_name
from reps.reps import REPS
from reps.ac_reps import ACREPS
from ppo.ppo import PPO

# Prepare and clean up arguments
args = ArgumentParser.parse()
args.env = gym.make(get_env_name(args.env, sim=not args.robot))
if args.seed is not None:
    args.env.seed(args.seed)
    print(f'# Seeded Env. Seed={args.seed}')
pprint(vars(args))

algos = {
    'REPS': REPS,
    'ACREPS': ACREPS,
    'PPO': PPO,
}
model = algos[args.algo_name](**vars(args))
model.train()
