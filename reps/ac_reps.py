import os

import torch
import autograd.numpy as np
from autograd import grad, jacobian
from scipy.optimize import minimize

from tensorboardX import SummaryWriter


class ACREPS:
    """
    Actor-Critic Relative Entropy Policy Search based on
    https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12247
    """

    def __init__(self, *, name, env, n_epochs=50, n_steps=3000, gamma=0.99, epsilon=0.1, n_fourier=75,
                 fourier_band=None, render=False, resume=False, eval=False, seed=None, summary_path=None,
                 checkpoint_path=None, **kwargs):
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

        self.γ = gamma
        self.ε = epsilon

        fourier_dim = self.env.observation_space.shape[0]

        if self.resume or self.eval:
            self.load_model()
        else:
            if fourier_band is None:
                # if fourier_band is not set, then set it heuristically.
                fourier_band = np.clip(self.env.observation_space.high, -10, 10) / 2.0
            fourier_band = list(fourier_band)
            fourier_cov = np.eye(len(fourier_band)) / fourier_band
            self.fourier_freq = np.random.multivariate_normal(np.zeros_like(fourier_band), fourier_cov, n_fourier)
            self.fourier_offset = 2 * np.pi * np.random.rand(n_fourier, fourier_dim) - np.pi
            self.θ = np.random.randn(n_fourier, self.env.action_space.shape[0])
            self.Σ = np.array([[16.0**2]])  # TODO: Adapt dimension to environment
            self.α = np.random.randn(n_fourier)
            self.η = np.random.rand()
            self.epoch = 0

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

                # action = np.random.multivariate_normal(mean=φ_s.T@self.θ, cov=self.Σ)
                action = self.π(φ_s)
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
                return η * self.ε + max_δ + η * np.log(np.mean(np.exp((δ - max_δ) / η))) + np.mean(V_s) \
                       + 1e-9 * np.linalg.norm(α, 2)

            params = np.concatenate([np.array([self.η]), self.α])
            bounds = [(1e-8, None)] + [(None, None)] * len(self.α)  # bounds for η and α
            res = minimize(dual, params, method='SLSQP', jac=jacobian(dual), bounds=bounds)
            self.η, self.α = res.x[0], res.x[1:]

            ############################
            #      FIT NEW POLICY      #
            ############################
            δ = returns - self.α.dot(φ.T)
            ω = np.expand_dims(np.exp(δ / self.η), axis=-1)
            # The KL can be computed by looking at the weights ω only
            ω_ = ω / np.mean(ω)
            kl = np.mean(ω_ * np.log(ω_))

            W = np.eye(len(ω)) * ω
            Φ = np.array(states)
            a = np.array(actions)

            # Update policy parameters
            z = (np.square(np.sum(ω)) - np.sum(np.square(ω))) / np.sum(ω)
            self.θ = np.linalg.solve(Φ.T @ W @ Φ + 1e-9 * np.eye(Φ.shape[-1]), Φ.T @ W @ a)
            self.Σ = np.eye(self.env.action_space.shape[0]) * np.sum(W @ np.square(a - Φ @ self.θ), axis=0) / z
            self.save_model()

            # Evaluate the deterministic policy
            n_eval_traj = 25
            _, mean_traj_reward = self.evaluate(n_eval_traj)
            policy_entropy = normal_entropy(self.Σ)
            self.save_tb_summary(self.epoch, rewards, mean_traj_reward, policy_entropy, self.η, kl)
            print(f"{self.epoch:4} rewards {np.sum(rewards):13.6f} - {mean_traj_reward:13.6f} |"
                  f"KL {kl:8.6f} | Σ {self.Σ} | entropy {policy_entropy}")

    def evaluate(self, n_trajectories):
        """
        Evaluate the deterministic policy for N full trajectories.
        :param n_trajectories: number of trajectories to use for the evaluation.
        :return (cumulative_reward, mean_traj_reward):
        """
        print('EVALUATE for', n_trajectories, 'trajectories')
        total_reward, trajectory = 0, 0
        Σ = np.zeros_like(self.Σ)
        φ_s = self.φ_fn(self.env.reset())
        while trajectory < n_trajectories:
            # action = np.random.multivariate_normal(mean=φ_s.T @ self.θ, cov=Σ)
            action = self.π(φ_s, deterministic=True)
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            if self.render:
                self.env.render()
            if done:
                trajectory += 1
                φ_s = self.φ_fn(self.env.reset())
            else:
                φ_s = self.φ_fn(next_state)
        mean_traj_reward = total_reward / n_trajectories
        return total_reward, mean_traj_reward

    def π(self, φ_s, deterministic=False):
        """
        Gaussian policy.
        :param φ_s:
        :param deterministic:
        :return:
        """
        if not deterministic:
            return np.random.multivariate_normal(mean=φ_s.T @ self.θ, cov=self.Σ)
        else:
            return φ_s.T @ self.θ

    def φ_fn(self, state):
        """
        Calculates the fourier feature vector of a given environment state.
        :param state: environment state
        :return: feature vector for state
        """
        return np.sin(np.sum(self.fourier_freq * (state + self.fourier_offset), axis=-1))

    def initialize_model(self):
        pass  # TODO: smart initialization for model parameters.

    def load_model(self):
        """Loads the last checkpoint to continue training or for evaluation."""
        file = np.load(self.checkpoint_path)
        self.epoch = file['epoch']
        self.fourier_freq, self.fourier_offset = file['fourier_features']  # feature parameters
        self.θ, self.Σ = file['θ'], file['Σ']  # policy parameters
        self.α, self.η = file['α'], file['η']  # dual parameters
        print(f"LOADED Model at epoch {self.epoch}")

    def save_model(self):
        """Saves a checkpoint of the current model."""
        np.savez(self.checkpoint_path,
                 epoch=self.epoch,
                 θ=self.θ, α=self.α,  # policy parameters
                 η=self.η, Σ=self.Σ,  # dual parameters
                 fourier_features=(self.fourier_freq, self.fourier_offset))  # feature parameters

    def save_tb_summary(self, epoch, rewards, mean_traj_reward, entropy, η, kl):
        self.writer.add_scalar('rl/reward', torch.tensor(np.sum(rewards), dtype=torch.float32), epoch)
        self.writer.add_scalar('rl/mean_traj_reward', mean_traj_reward, epoch)
        self.writer.add_scalar('rl/entropy', torch.tensor(entropy, dtype=torch.float32), epoch)
        self.writer.add_scalar('rl/η', torch.tensor(η), epoch)
        self.writer.add_scalar('rl/KL', torch.tensor(kl), epoch)

def normal_entropy(Σ):
    """Calculates the entropy of a gaussian policy given it's covariance matrix Σ."""
    return 0.5 * np.log(2*np.e*np.pi*np.linalg.det(Σ))

