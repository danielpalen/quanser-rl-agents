import torch
import numpy as np
from scipy.optimize import minimize

from tensorboardX import SummaryWriter

class ACREPS:
    """
    Actor-Critic Relative Entropy Policy Search based on
    https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12247
    """

    def __init__(self, name, env, n_epochs, n_steps, gamma, epsilon, n_fourier, fourier_band, render=False,
                 resume=False, eval=False, seed=None, **kwargs):

        self.name = name
        self.env = env
        self.n_epochs = n_epochs
        self.n_steps = n_steps
        self.render = render
        self.resume = resume
        self.eval = eval
        self.training = not eval
        self.writer = SummaryWriter(log_dir=f"./out/summary/{self.name}")

        self.γ = gamma
        self.ε = epsilon

        fourier_dim = self.env.observation_space.shape[0]

        if self.resume or self.eval:
            file = np.load(f"./out/models/{self.name}.npz")
            self.fourier_features = file['fourier_features']
            self.θ = file['θ']
            self.σ = file['σ'] if self.resume else 0.0  # zero variance for evaluation
            self.α = file['α']
            self.η = file['η']
            self.epoch = file['epoch']
            print(f"LOADED Model at epoch {self.epoch}")
        else:
            fourier_feature_parameters = []
            fourier_cov = np.eye(len(fourier_band)) / fourier_band
            for _ in range(n_fourier):
                freq = np.random.multivariate_normal(np.zeros_like(fourier_band), fourier_cov)
                offset = 2 * np.pi * np.random.rand(fourier_dim) - np.pi
                fourier_feature_parameters.append((freq, offset))
            self.fourier_features = np.array(fourier_feature_parameters)
            self.θ = np.random.randn(n_fourier, self.env.action_space.shape[0])
            self.σ = 16.0
            self.α = np.random.randn(n_fourier)
            self.η = np.random.rand()
            self.epoch = 0
        self.kl = 0.0

    def φ_fn(self, state):
        """
        Calculates the feature vector of a given state using the self.fourier_features.
        :param state: environment state
        :return: feature vector for state
        """
        feature_vector = [np.sin(f @ (state + o)) for f, o in self.fourier_features]
        return np.array(feature_vector)

    def train(self):
        while self.epoch < self.n_epochs or self.eval:
            ############################
            #        SAMPLING          #
            ############################

            self.epoch += 1
            states, next_states, actions, rewards, dones = [], [], [], [], []

            state = self.env.reset()

            # Sample trajectories
            for step in range(self.n_steps):

                # action = np.random.multivariate_normal(mean=φ_fn(state).T @ θ, cov=σ)
                action = np.random.normal(loc=self.φ_fn(state).T @ self.θ, scale=self.σ)
                state_, reward, done, _ = self.env.step(action)

                if self.render:
                    self.env.render()

                states.append(self.φ_fn(state))
                actions.append(action)
                rewards.append(reward)
                next_states.append(self.φ_fn(state_))
                dones.append(done)

                if done:
                    state = self.env.reset()
                else:
                    state = state_

                print(f"step {step}", end="\r")

            if self.training:

                ############################
                #     DUAL OPTIMIZATION    #
                ############################

                rewards = np.array(rewards)
                returns = np.zeros_like(rewards)
                R = 0
                for t in reversed(range(len(rewards))):
                    R = rewards[t] + self.γ * R * (1 - dones[t])
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
                    δ = returns - α.dot(φ.T)
                    return η * self.ε + np.max(δ) + η * np.log(np.mean(np.exp((δ - np.max(δ)) / η))) + np.mean(
                        α.dot(φ.T)) + 1e-9 * np.linalg.norm(α, 2)

                params = np.concatenate([np.array([self.η]), self.α])
                bounds = [(1e-8, None)] + [(None, None)] * len(self.α)  # bounds for η and α
                res = minimize(dual, params, method='SLSQP', bounds=bounds)
                self.η, self.α = res.x[0], res.x[1:]

                ############################
                #      FIT NEW POLICY      #
                ############################

                δ = returns - self.α.dot(φ.T)
                ω = np.expand_dims(np.exp(δ / self.η), axis=-1)
                ω_ = ω / np.mean(ω)
                self.kl = np.mean(ω_ * np.log(ω_))

                W = np.eye(len(ω)) * ω
                Φ = np.array(states)
                a = np.array(actions)

                # Update policy parameters
                self.θ = np.linalg.solve(Φ.T @ W @ Φ + 1e-9 * np.eye(Φ.shape[-1]), Φ.T @ W @ a)
                Z = (np.square(np.sum(ω)) - np.sum(np.square(ω))) / np.sum(ω)
                # σ = np.sqrt(np.sum(W @ np.square(ã - Φ @ θ)) / Z)
                self.σ = 0.5 * self.σ + 0.5 * np.sqrt(np.sum(W @ np.square(a - Φ @ self.θ)) / Z)
                # σ = np.eye(env.action_space.shape[0]) * np.sum(W @ np.square(ã - Φ @ θ), axis=0) / Z

                self.writer.add_scalar('rl/reward', torch.tensor(np.sum(rewards) / np.sum(dones), dtype=torch.float32), self.epoch)
                self.writer.add_scalar('rl/η', torch.tensor(self.η), self.epoch)
                self.writer.add_scalar('rl/KL', torch.tensor(self.kl), self.epoch)

                np.savez(f"./out/models/{self.name}.npz", θ=self.θ, α=self.α, η=self.η, σ=self.σ,
                         fourier_features=self.fourier_features, epoch=self.epoch)

            print(f"{self.epoch:4} rewards {np.sum(rewards) / np.sum(dones):13.6f} | KL {self.kl:8.6f} | σ {self.σ}")
