import os

import autograd.numpy as np
from autograd import jacobian
from scipy.optimize import minimize

from tensorboardX import SummaryWriter

from common.common import save_tb_scalars, gaussian_policy as π


class ACREPS:

    def __init__(self, *, name, env, n_epochs=50, n_steps=3000, gamma=0.99, epsilon=0.1, sigma=16.0, n_fourier=75,
                 fourier_band=None, render=False, resume=False, eval=False, seed=None, summary_path=None,
                 checkpoint_path=None, **kwargs):
        """
        Actor-Critic Relative Entropy Policy Search based on
        https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12247

        :param name: experiment name for checkpointing
        :param env: instance of OpenAI gym environment
        :param n_epochs: number of training epochs
        :param n_steps: number of environment steps per training epoch
        :param gamma: discount factor used when calculating the return.
        :param epsilon: Kullback-Leibler divergence bound ε.
        :param sigma: standard deviation of the current policy.
        :param n_fourier: number of fourier features to used for the model.
        :param fourier_band: bandwidth to use for the covariance matrix of the fourier features.
        :param render: renders the environment.
        :param resume: loads the last checkpoint to continue to train.
        :param eval: loads the last checkpoint to perform evaluation of the deterministic policy afterwards..
        :param seed: optional seed.
        :param summary_path: path at which tensorboard summary files are saved.
        :param checkpoint_path: path at which model checkpoints are saved and loaded.
        :param kwargs: Helper to catch unused arguments supplied by the argument parser in run.py
        """

        if seed is not None:
            np.random.seed(seed)

        self.name = name
        self.env = env
        self.n_epochs = n_epochs
        self.n_steps = n_steps
        self.render = render
        self.resume = resume
        self.eval = eval
        self.summary_path = summary_path
        self.checkpoint_path = os.path.join(checkpoint_path, self.name + '.npz')
        self.writer = SummaryWriter(log_dir=self.summary_path)

        self.epoch = 0
        self.γ = gamma
        self.ε = epsilon

        # These will either be loaded from a checkpoint or initialized right after.
        self.fourier_freq, self.fourier_offset = None, None  # feature parameters
        self.θ, self.Σ = None, None  # policy parameters
        self.α, self.η = None, None  # dual parameters
        if self.resume or self.eval:
            self.load_model()
        else:
            self.initialize_model(n_fourier, fourier_band, sigma)

    def train(self):
        """
        Training loop for the ACREPS model
        """
        while self.epoch < self.n_epochs:
            ############################
            #        SAMPLING          #
            ############################
            self.epoch += 1
            states, actions, rewards, dones = [], [], [], []
            φ_s = self.φ_fn(self.env.reset())

            # Sample trajectories
            for step in range(self.n_steps):
                action = π(φ_s, θ=self.θ, Σ=self.Σ)
                next_state, reward, done, _ = self.env.step(action)
                states.append(φ_s)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                if self.render:
                    self.env.render()
                if done:
                    φ_s = self.φ_fn(self.env.reset())
                else:
                    φ_s = self.φ_fn(next_state)
                print(f"step {step}", end="\r")

            ############################
            #     DUAL OPTIMIZATION    #
            ############################
            φ = np.array(states)
            rewards = np.array(rewards)
            returns = np.zeros_like(rewards)
            R = 0
            for t in reversed(range(len(rewards))):
                R = rewards[t] + self.γ * R * (1 - dones[t])
                returns[t] = R

            def dual(p):
                """dual formulation of the ACREPS objective function"""
                η, α = p[0], p[1:]
                V_s = np.dot(α, φ.T)
                δ = returns - V_s
                max_δ = np.max(δ)
                return η*self.ε + max_δ + η*np.log(np.mean(np.exp((δ-max_δ)/η))) + np.mean(V_s) + 1.e-16*np.linalg.norm(α, 2)

            params = np.concatenate([np.array([self.η]), self.α])
            bounds = [(1e-8, None)] + [(None, None)] * len(self.α)  # bounds for η and α
            res = minimize(dual, params, method='SLSQP', jac=jacobian(dual), bounds=bounds)
            self.η, self.α = res.x[0], res.x[1:]

            ############################
            #      FIT NEW POLICY      #
            ############################
            δ = returns - self.α.dot(φ.T)
            ω = np.exp(δ / self.η)
            W = np.diag(ω)
            Φ = np.array(states)
            a = np.array(actions)
            # Update policy parameters
            z = (np.square(np.sum(ω)) - np.sum(np.square(ω))) / np.sum(ω)
            self.θ = np.linalg.solve(Φ.T @ W @ Φ + 1.e-16 * np.eye(Φ.shape[-1]), Φ.T @ W @ a)
            self.Σ = np.eye(self.env.action_space.shape[0]) * np.sum(W @ np.square(a - Φ @ self.θ), axis=0) / z
            self.save_model()

            # The KL can be computed by looking at the weights ω only
            ω_ = ω / np.mean(ω)
            kl = np.mean(ω_ * np.log(ω_))

            if self.epoch%10==0:
                # Evaluate the deterministic policy
                n_eval_traj = 25
                _, mean_traj_reward = self.evaluate(n_eval_traj)
                entropy = normal_entropy(self.Σ)
                save_tb_scalars(self.writer, self.epoch, reward=np.sum(rewards), mean_traj_reward=mean_traj_reward,
                                entropy=entropy, η=self.η, kl=kl)
            else:
                entropy = normal_entropy(self.Σ)
                save_tb_scalars(self.writer, self.epoch, reward=np.sum(rewards), entropy=entropy, η=self.η, kl=kl)

    def evaluate(self, n_trajectories, print_reward=False):
        """
        Evaluate the deterministic policy for N full trajectories.
        :param n_trajectories: number of trajectories to use for the evaluation.
        :return (cumulative_reward, mean_traj_reward):
        """
        total_reward, trajectory = 0, 0
        traj_rewards = []
        traj_reward = 0
        state = self.env.reset()
        φ_s = self.φ_fn(state)
        step = 0

        actions = []
        states = []

        print('Evaluating the deterministic policy...')

        while len(traj_rewards) < n_trajectories:
            step += 1
            action = π(φ_s, θ=self.θ, Σ=self.Σ, deterministic=True)
            states.append(state)
            actions.append(action)
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            traj_reward += reward
            if self.render:
                self.env.render()
            if done:
                # print(step)
                step = 0
                φ_s = self.φ_fn(self.env.reset())
                trajectory += 1
                traj_rewards.append(traj_reward)
                traj_reward = 0
                if print_reward:
                    # print(traj_rewards)
                    print(len(traj_rewards), 'trajectories: total', total_reward, 'mean', np.mean(traj_rewards), 'std', np.std(traj_rewards),
                          'max', np.max(traj_rewards))
                    #print('states', states)
                    #print('actions', actions)
                    # print()
                states = []
                actions = []
            else:
                state = next_state
                φ_s = self.φ_fn(next_state)
        mean_traj_reward = total_reward / n_trajectories
        if print_reward:
            print('FINAL: total', total_reward, 'mean', np.mean(traj_rewards), 'std', np.std(traj_rewards), 'max', np.max(traj_rewards))
            print()
        return total_reward, mean_traj_reward

    def φ_fn(self, state):
        """
        Calculates the fourier feature vector of a given environment state.
        :param state: environment state
        :return: feature vector for state
        """
        return np.sin(np.sum(self.fourier_freq * (state + self.fourier_offset), axis=-1))

    def initialize_model(self, n_fourier, fourier_band, sigma):
        """Initialize model parameters."""
        if fourier_band is None:
            # if fourier_band is not set, then set it heuristically.
            fourier_band = np.clip(self.env.observation_space.high, -10, 10) / 2.0
        print('fourier_band', fourier_band)
        fourier_cov = np.linalg.inv(np.diag(fourier_band))
        fourier_dim = self.env.action_space.shape[0]
        self.fourier_freq = np.random.multivariate_normal(np.zeros_like(fourier_band), fourier_cov, n_fourier)
        self.fourier_offset = 2 * np.pi * np.random.rand(n_fourier, self.env.observation_space.shape[0]) - np.pi
        self.θ, self.Σ = np.random.randn(n_fourier, fourier_dim), np.eye(fourier_dim) * sigma ** 2
        self.α, self.η = np.random.randn(n_fourier), np.random.rand()

    def load_model(self):
        """Loads the last checkpoint to continue training or for evaluation."""
        file = np.load(self.checkpoint_path)
        self.epoch = file['epoch']
        self.fourier_freq, self.fourier_offset = file['fourier_features']  # feature parameters
        self.θ, self.Σ = file['θ'], [[16.0]]  # policy parameters
        self.α, self.η = file['α'], file['η']  # dual parameters
        print(f"LOADED Model at epoch {self.epoch}")

    def save_model(self):
        """Saves a checkpoint of the current model."""
        np.savez(self.checkpoint_path,
                 epoch=self.epoch,
                 θ=self.θ, α=self.α,  # policy parameters
                 η=self.η, Σ=self.Σ,  # dual parameters
                 fourier_features=(self.fourier_freq, self.fourier_offset))  # feature parameters


def normal_entropy(Σ):
    """Calculates the entropy of a gaussian policy given it's covariance matrix Σ."""
    return 0.5 * np.log(2*np.e*np.pi*np.linalg.det(Σ))

