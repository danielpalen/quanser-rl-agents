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
    'pendulum':        'Pendulum-v0',
}

env = gym.make(environments[args.env])

##############################
# Generate Fourier Features  #
##############################

feature_dim = env.observation_space.shape[0]
n_fourier_features = 75
band = np.array([0.5, 0.5, 4])
cov = np.eye(3) * 1/band

ε = 0.1
γ = 0.99
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
        freq = np.random.multivariate_normal(np.array([0, 0, 0]), cov)
        offset = 2 * np.pi * np.random.rand(feature_dim) - np.pi
        fourier_feature_parameters.append((freq, offset))
    fourier_feature_parameters = np.array(fourier_feature_parameters)

    θ = np.random.randn(n_fourier_features, env.action_space.shape[0])
    σ = 16.0
    α = np.random.randn(n_fourier_features)
    η = np.random.rand()
    epoch = 0


def φ_fn(s):
    """Feature function to generate fourier features from environment states."""
    feature_vector = [np.sin(f @ (s + o)) for f, o in fourier_feature_parameters]
    return np.array(feature_vector)


n_steps = args.n_steps

while epoch < 100:

    ############################
    #        SAMPLING          #
    ############################

    epoch += 1
    states, states_0, next_states, actions, rewards = [], [], [], [], []

    state = env.reset()
    states_0.append(φ_fn(state))

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

        if np.random.rand() < 1-γ:
            state = env.reset()
            states_0.append(φ_fn(state))
        else:
            state = state_

    if training:

        ############################
        #     DUAL OPTIMIZATION    #
        ############################

        R = np.array(rewards)
        φ = np.array(states)
        φ_ = np.array(next_states)
        φ_0 = np.expand_dims(np.mean(states_0, axis=0), axis=0)

        Φ = γ * φ_ - φ + (1 - γ) * φ_0

        def dual(p):
            """dual formulation for of the REPS objective function"""
            η, α = p[0], p[1:]
            δ = R + α.dot(Φ.T)
            return η * ε + np.max(δ) + η * np.log(np.mean(np.exp((δ-np.max(δ))/η))) + 1e-6 * np.linalg.norm(α, 2)

        params = np.concatenate([np.array([η]), α])
        bounds = [(1e-8, None)] + [(None, None)] * len(α)  # bounds for η and α
        res = minimize(dual, params, method='SLSQP', bounds=bounds)
        η, α = res.x[0], res.x[1:]

        ############################
        #      FIT NEW POLICY      #
        ############################

        δ = R + α.dot(Φ.T)
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

        writer.add_scalar('rl/reward', torch.tensor(np.mean(rewards)), epoch)
        writer.add_scalar('rl/η',  torch.tensor(η), epoch)
        writer.add_scalar('rl/KL', torch.tensor(KL), epoch)
        writer.add_scalar('rl/σ', torch.tensor(σ), epoch)

        np.savez(f"./out/models/{args.experiment_id}.npz", θ=θ, α=α, η=η, σ=σ,
                 fourier_feats=fourier_feature_parameters, epoch=epoch)

    print(f"{epoch:4} rewards {np.mean(rewards):10.6f} | KL {KL:8.6f} | σ {σ}")
