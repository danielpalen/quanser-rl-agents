import argparse

import torch
import numpy as np
from scipy.optimize import minimize

from tensorboardX import SummaryWriter

import gym
import quanser_robots

parser = argparse.ArgumentParser(description='Solve the different gym envs with PPO')
parser.add_argument('experiment_id', type=str, help='identifier to store experiment results')
parser.add_argument('--env', type=str, default='pendulum',
                    help="environment to be used for training [pendulum, double_pendulum, furuta, balancer]")
parser.add_argument('--eval', action='store_true', help='toggles evaluation mode')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--resume', action='store_true',
                    help='resume training on an existing model by loading the last checkpoint')
parser.add_argument('--n_steps', type=int, default=3000,
                    help='number of agent steps when collecting trajectories for one epoch')

args = parser.parse_args()
writer = SummaryWriter(log_dir=f"./out/summary/{args.experiment_id}")

training = not args.eval


environments = {
    'balancer':        'BallBalancerSim-v0',
    'double_pendulum': 'DoublePendulum-v0',
    'furuta':          'Qube-v0',
    'furutaRR': 'QubeRR-v0',
    'pendulum':        'Pendulum-v0',
}

env = gym.make(environments[args.env])
print('Env:', env)
print('Reward range', env.reward_range)
print('Observation space:', env.observation_space)
print('  ', env.observation_space.high)
print('  ', env.observation_space.low)
print('Action space:', env.action_space)
print('  ', env.action_space.high)
print('  ', env.action_space.low)

##############################
# Generate Fourier Features  #
##############################

feature_dim = env.observation_space.shape[0]
n_fourier_features = 75
# band = np.array([0.5, 0.5, 4])
# band = np.array([0.1, 0.1, 0.1, 10., 10., 10.])
band = np.array([0.5, 0.5, 0.5, 0.5, 5., 5.])
cov = np.eye(len(band)) * 1/band

ε = 0.1
γ = 0.95
KL = 0.0

if args.resume or args.eval:
    file = np.load(f"./out/models/{args.experiment_id}.npz")
    fourier_feature_parameters = file['fourier_feats']
    θ = file['θ']
    σ = file['σ'] if args.resume else 0.0  # zero variance for evaluation
    α = file['α']
    η = file['η']
    epoch = file['epoch']
    print(f"LOADED Model at epoch {epoch}")
else:
    fourier_feature_parameters = []
    for _ in range(n_fourier_features):
        freq = np.random.multivariate_normal(np.zeros_like(band), cov)
        offset = 2 * np.pi * np.random.rand(feature_dim) - np.pi
        fourier_feature_parameters.append((freq, offset))
    fourier_feature_parameters = np.array(fourier_feature_parameters)

    θ = np.random.randn(n_fourier_features, env.action_space.shape[0])
    σ = 16.0
    # σ = 32.0
    # σ = 64.0
    α = np.random.randn(n_fourier_features)
    η = np.random.rand()
    epoch = 0


def φ_fn(s):
    """Feature function to generate fourier features from environment states."""
    feature_vector = [np.sin(f @ (s + o)) for f, o in fourier_feature_parameters]
    return np.array(feature_vector)


n_steps = args.n_steps

while epoch < 10000:

    ############################
    #        SAMPLING          #
    ############################

    epoch += 1
    states, next_states, actions, rewards, dones = [], [], [], [], []

    state = env.reset()

    # Sample trajectories
    for step in range(n_steps):

        action = np.random.normal(loc=φ_fn(state).T @ θ, scale=σ)
        state_, reward, done, _ = env.step(action)

        if args.render:
            env.render()

        states.append(φ_fn(state))
        actions.append(action)
        rewards.append(reward)
        next_states.append(φ_fn(state_))
        dones.append(done)

        if done:
            state = env.reset()
        else:
            state = state_

        print(f"step {step}", end="\r")

    if training:

        ############################
        #     DUAL OPTIMIZATION    #
        ############################

        rewards = np.array(rewards)
        returns = np.zeros_like(rewards)
        R = 0
        for t in reversed(range(len(rewards))):
            R = rewards[t] + γ * R * (1-dones[t])
            returns[t] = R

        # for r, ret, d in zip(rewards, returns, dones):
        #     print(d, r, ret)

        φ = np.array(states)
        # φ_ = np.array(next_states)

        # Φ = γ * φ_ - φ + (1 - γ) * φ_0

        # print('returns', returns.shape)

        def dual(p):
            """dual formulation of the ACREPS objective function"""
            η, α = p[0], p[1:]
            #print('α', α.shape)
            #print('φ', φ.shape)
            #print('α.dot(φ)', (α.dot(φ.T)).shape)
            δ = returns - α.dot(φ.T)
            #print('δ', δ.shape)
            return η * ε + np.max(δ) + η * np.log(np.mean(np.exp((δ-np.max(δ))/η))) + np.mean(α.dot(φ.T)) + 1e-9 * np.linalg.norm(α, 2)

        params = np.concatenate([np.array([η]), α])
        bounds = [(1e-8, None)] + [(None, None)] * len(α)  # bounds for η and α
        res = minimize(dual, params, method='SLSQP', bounds=bounds)
        η, α = res.x[0], res.x[1:]

        ############################
        #      FIT NEW POLICY      #
        ############################

        #δ = R + α.dot(Φ.T)
        δ = returns - α.dot(φ.T)
        ω = np.expand_dims(np.exp(δ / η), axis=-1)
        ω_ = ω / np.mean(ω)
        KL = np.mean(ω_*np.log(ω_))

        W = np.eye(len(ω)) * ω
        Φ = np.array(states)
        ã = np.array(actions)

        # Update policy parameters
        θ = np.linalg.solve(Φ.T @ W @ Φ + 1e-6 * np.eye(Φ.shape[-1]), Φ.T @ W @ ã)

        Z = (np.square(np.sum(ω)) - np.sum(np.square(ω))) / np.sum(ω)
        σ = np.sqrt(np.sum(W @ np.square(ã - Φ @ θ)) / Z)

        writer.add_scalar('rl/reward', torch.tensor(np.mean(rewards), dtype=torch.float32), epoch)
        writer.add_scalar('rl/η',  torch.tensor(η), epoch)
        writer.add_scalar('rl/KL', torch.tensor(KL), epoch)
        writer.add_scalar('rl/σ', torch.tensor(σ), epoch)

        np.savez(f"./out/models/{args.experiment_id}.npz", θ=θ, α=α, η=η, σ=σ,
                 fourier_feats=fourier_feature_parameters, epoch=epoch)

    print(f"{epoch:4} rewards {np.mean(rewards):10.6f} | KL {KL:8.6f} | σ {σ}")
