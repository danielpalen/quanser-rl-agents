import sys

from pprint import pprint

import argparse
# import numpy as np
from scipy.optimize import minimize

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, MultivariateNormal

import numpy as np

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
n_fourier_features = 75

band = np.array([0.5,0.5,4])
cov = np.eye(3) * 1/band

fourier_feature_parameters = []
for _ in range(n_fourier_features):
    # freq   = np.random.randn(feature_dim)
    freq = np.random.multivariate_normal(np.array([0,0,0]), cov)
    offset = 2*np.pi*np.random.rand(feature_dim)-np.pi
    fourier_feature_parameters.append((freq, offset))
fourier_feature_parameters = np.array(fourier_feature_parameters)

print('fourier_feature_parameters', fourier_feature_parameters.shape)

def φ(s):
    """Feature function to generate fourier featutres from environment states."""
    feature_vector = [np.sin(f @ (s + o)) for f,o in fourier_feature_parameters]
    return np.array(feature_vector)

# Policy parameters θ
θ = np.random.randn(n_fourier_features, env.action_space.shape[0])
σ = 0.5
α = np.random.randn(n_fourier_features)
η = np.random.rand()
ε = 0.1
γ = 0.99

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
    state_0 = φ(state)
    states, states_0, next_states, actions, old_probs, rewards, values, dones = [],[],[],[],[],[],[],[]

    # Sample trajectories
    for step in range(n_steps):

        μ = φ(state).T @ θ
        a_dist = Normal(torch.tensor(μ),σ)#torch.from_numpy(μ), σ)
        action = a_dist.sample()#.squeeze(0)
        old_prob = a_dist.log_prob(action)

        # print('μ',μ)
        # print('a', action)
        # print('l', old_prob)
        # print()

        state_, reward, done, _ = env.step(action) #env.step(action)

        if args.render:
            env.render()

        states.append(φ(state))
        states_0.append(state_0)
        actions.append(action.numpy())
        rewards.append(reward)
        next_states.append(φ(state_))
        dones.append(done)
        old_probs.append(old_prob.numpy())

        # state = state_

        if np.random.rand() < 1-γ:
            # print('reset')
            state = env.reset()
            state_0 = φ(state)
        else:
            state = state_

        # if done:
        #     state = env.reset()
        # else:
        #     state = state_


    # print('actions', actions)
    #print('old_probs', old_probs)
    #

    ############################
    #                          #
    #     DUAL OPTIMIZATION    #
    #                          #
    ############################

    def dual(params):
        """dual formulation for of the REPS objective function"""
        η,α = params[0], params[1:]
        φ_0 = np.mean(state_0, axis=-1)
        δ = np.array([(r + α.dot( γ*φ_s_ - φ_s + (1-γ)*φ_0 ))
                      for r, φ_s_, φ_s in zip(rewards, next_states, states)])
        return η*ε + np.max(δ) + η * np.log(np.mean(np.exp((δ-np.max(δ))/η)))

    # print('inital dual value', dual(np.concatenate([np.array([η]),α])))

    params = np.concatenate([np.array([η]),α])
    bounds = [(1e-8,None)] + [(None,None)]*len(α) # bounds for η and α
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

    φ_0 = np.mean(state_0, axis=-1)
    δ = np.array([(r + α.dot( γ*φ_s_ - φ_s + (1-γ)*φ_0 )) for r, φ_s, φ_s_ in zip(rewards, states, next_states)])
    w = np.expand_dims(np.exp(δ / η), axis=-1)
    # print('w_', np.mean(w))
    w = w / np.mean(w)

    #print('w',w)
    # print('w', w)
    # print('w_', np.mean(w))
    print('KL', np.mean(w*np.log(w)))

    #print('w', w.shape)

    D = np.eye(len(w)) * w
    Φ = np.array(states)
    ã = np.array(actions)

    # print('D', D)
    # print('Φ', Φ.shape)
    # print('ã', ã.shape)


    #a_ = w * np.array(actions)
    #φ_ = w * np.array(states)
    # print('φ_', φ_.shape)

    θ = np.linalg.solve(Φ.T @ D @ Φ, Φ.T @ D @ ã)

    # print('ã', ã.shape)
    # print('Φ', Φ.shape)
    # print('D', D.shape)
    # print('θ', θ.shape)
    #
    #
    # print('zähler', np.sum( np.square( ã - D @ Φ @ θ ) ))
    # print('nenner', np.trace(D))
    Z = ( np.square(np.sum(w)) - np.sum(np.square(w)) ) / np.sum(w)
    σ = np.sqrt( np.sum( D @ np.square(ã - Φ @ θ) ) / Z )

    # print('θ', θ.shape)
    print('σ', σ)

    # if epoch%10==0:
    print(f"{epoch:4} rewards {np.mean(rewards):10.6f}")
    print('###################################################################')
