import sys

from pprint import pprint

import argparse
import numpy as np
from scipy.optimize import minimize

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal, MultivariateNormal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import gym
import quanser_robots


parser = argparse.ArgumentParser(description='Solve the different gym envs with PPO')
# parser.add_argument('experiment_id', type=str, help='identifier to store experiment results')
parser.add_argument('--env', type=str, default='pendulum', help="name of the gym environment to be used for training [pendulum, double_pendulum, furuta, balancer]")
parser.add_argument('--eval', action='store_true', help='toggles evaluation mode')
parser.add_argument('--render', action='store_true', help='render the environment')
# parser.add_argument('--resume', action='store_true', help='resume training on an existing model by loading the last checkpoint')

parser.add_argument('--gamma', type=float, default=0.9, help='discount factor')
parser.add_argument('--n_steps', type=int, default=1000, help='number of agent steps when collecting trajectories for one epoch')

args = parser.parse_args()

torch.set_printoptions(threshold=5000)

training = not args.eval


environments = {
    'balancer'        : 'BallBalancerSim-v0',
    'double_pendulum' : 'DoublePendulum-v0',
    'furuta'          : 'Qube-v0',
    'pendulum'        : 'Pendulum-v0',
}


env = gym.make(environments[args.env])


##############################
#                            #
# Generate Fourier Features  #
#                            #
##############################

feature_dim = env.observation_space.shape[0]
n_fourier_features = 100

fourier_feature_parameters = []
for _ in range(n_fourier_features):
    freq   = np.random.randn(feature_dim)
    offset = 2*np.pi*np.random.rand(feature_dim)-np.pi
    fourier_feature_parameters.append((freq, offset))
fourier_feature_parameters = np.array(fourier_feature_parameters)

print('fourier_feature_parameters', fourier_feature_parameters.shape)

def φ(s):
    """Feature function to generate fourier featutres from environment states."""
    feature_vector = [np.sin(f @ (s + o)) for f,o in fourier_feature_parameters]
    return np.array(feature_vector)

# Policy parameters θ
θ = np.random.randn(env.action_space.shape[0], n_fourier_features)
α = np.random.randn(n_fourier_features)
η = np.random.rand()
ε = 1e-12

print('θ', θ.shape)
print('η', η, 'α', α, 'ε', ε )

epoch = 0
n_steps = args.n_steps


while epoch < 1000:

    ############################
    #                          #
    #        SAMPLING          #
    #                          #
    ############################

    epoch += 1
    state = env.reset()
    states, next_states, actions, log_probs, rewards, values, dones = [],[],[],[],[],[],[]

    # Sample trajectories
    for step in range(n_steps):

        μ,σ = θ.dot(φ(state)), 0.2
        action = np.random.normal(loc=μ, scale=σ)
        state_, reward, done, _ = env.step(action) #env.step(action)

        if args.render:
            env.render()

        states.append(φ(state))
        actions.append(action)
        rewards.append(reward)
        next_states.append(φ(state_))
        dones.append(done)

        # state = state_

        if np.random.rand() < 0.01:
            print('reset')
            state = env.reset()
        else:
            state = state_

        # if done:
        #     state = env.reset()
        # else:
        #     state = state_


    ############################
    #                          #
    #     DUAL OPTIMIZATION    #
    #                          #
    ############################

    def dual(params):
        """dual formulation for of the REPS objective function"""
        η,α = params[0], params[1:]
        δ = np.array([(r + α.dot( φ_s - φ_s_ ))
                      for r, φ_s, φ_s_ in zip(rewards, states, next_states)])
        return η*ε + η * np.log(np.mean(np.exp(δ/η)))

    # print('inital dual value', dual(np.concatenate([np.array([η]),α])))

    params = np.concatenate([np.array([η]),α])
    bounds = [(1e-5,None)] + [(None,None)]*len(α) # bounds for η and α
    res = minimize(dual, params, method='SLSQP', bounds=bounds)
    η,α = res.x[0], res.x[1:]

    params = np.concatenate([np.array([η]),α])
    # print('final dual value', dual(params))
    # print('η', η, ' α', α)


    ############################
    #                          #
    #      FIT NEW POLICY      #
    #                          #
    ############################

    δ = np.array([(r + α.dot( φ_s - φ_s_ )) for r, φ_s, φ_s_ in zip(rewards, states, next_states)])
    w = np.expand_dims(np.exp(δ / (2*η)), axis=-1) # /2η == sqrt of the whole thing.

    #print('w', w.shape)

    a_ = w * np.array(actions)
    φ_ = w * np.array(states)
    #print('φ_', φ_.shape)

    θ = np.linalg.inv(φ_.T.dot(φ_)).dot(φ_.T).dot(a_)
    θ = np.transpose(θ)

    print('θ', θ)

    # break

    #if epoch%10==0:
    print(f"{epoch:4} rewards {np.sum(rewards):10.2f}")
    print('###################################################################')
