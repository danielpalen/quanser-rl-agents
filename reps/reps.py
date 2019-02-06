import torch
import numpy as np
# import autograd.numpy as np
# from autograd import grad
from scipy.optimize import minimize

from tensorboardX import SummaryWriter


class REPS:
    """
    Relative Entropy Policy Search based on
    https://www.ias.informatik.tu-darmstadt.de/uploads/Team/JanPeters/Peters2010_REPS.pdf
    """

    def __init__(self, name, env, n_epochs=50, n_steps=3000, gamma=0.99, epsilon=0.1, n_fourier=75,
                 fourier_band=None, render=False, resume=False, eval=False, seed=None, **kwargs):
        """
        :param name:
        :param env:
        :param n_epochs:
        :param n_steps:
        :param render:
        :param resume:
        :param eval:
        :param gamma:
        :param epsilon:
        :param n_fourier:
        :param fourier_band:
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
            if fourier_band is None:
                # if fourier_band is not set, then set it heuristically.
                fourier_band = np.clip(self.env.observation_space.high, -10, 10) / 2.0
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

        print('fourier_band', fourier_band)

    def φ_fn(self, state):
        """
        Calculates the feature vector of a given state using the self.fourier_features.
        :param state: environment state
        :return: feature vector for state
        """
        feature_vector = [np.sin(f @ (state + o)) for f, o in self.fourier_features]
        return np.array(feature_vector)

    def train(self):
        """
        Train REPS
        """
        while self.epoch < self.n_epochs or self.eval:

            ############################
            #        SAMPLING          #
            ############################

            self.epoch += 1
            states, states_0, next_states, actions, rewards = [], [], [], [], []

            state = self.env.reset()
            states_0.append(self.φ_fn(state))

            # Sample trajectories
            for step in range(self.n_steps):

                action = np.random.normal(loc=self.φ_fn(state).T @ self.θ, scale=self.σ)
                state_, reward, done, _ = self.env.step(action)

                if self.render:
                    self.env.render()

                states.append(self.φ_fn(state))
                actions.append(action)
                rewards.append(reward)
                next_states.append(self.φ_fn(state_))

                if np.random.rand() < 1 - self.γ:
                    state = self.env.reset()
                    states_0.append(self.φ_fn(state))
                else:
                    state = state_
                print(f"step {step}", end="\r")

            if self.training:
                ############################
                #     DUAL OPTIMIZATION    #
                ############################

                R = np.array(rewards)
                φ = np.array(states)
                φ_ = np.array(next_states)
                φ_0 = np.expand_dims(np.mean(states_0, axis=0), axis=0)
                Φ = self.γ * φ_ - φ + (1 - self.γ) * φ_0

                def dual(p):
                    """dual formulation for of the REPS objective function"""
                    η, α = p[0], p[1:]
                    δ = R + α.dot(Φ.T)
                    return η * self.ε + np.max(δ) + η * np.log(np.mean(np.exp((δ - np.max(δ)) / η))) + 1e-6 * np.linalg.norm(α, 2)

                params = np.concatenate([np.array([self.η]), self.α])
                bounds = [(1e-8, None)] + [(None, None)] * len(self.α)  # bounds for η and α
                res = minimize(dual, params, method='SLSQP', bounds=bounds)
                self.η, self.α = res.x[0], res.x[1:]

                ############################
                #      FIT NEW POLICY      #
                ############################

                δ = R + self.α.dot(Φ.T)
                ω = np.expand_dims(np.exp(δ / self.η), axis=-1)
                # The KL can be computed by looking at the weights only
                ω_ = ω / np.mean(ω)
                self.kl = np.mean(ω_ * np.log(ω_))

                W = np.eye(len(ω)) * ω
                Φ = np.array(states)
                a = np.array(actions)

                # Update policy parameters
                self.θ = np.linalg.solve(Φ.T @ W @ Φ + 1e-6 * np.eye(Φ.shape[-1]), Φ.T @ W @ a)

                Z = (np.square(np.sum(ω)) - np.sum(np.square(ω))) / np.sum(ω)
                self.σ = np.sqrt(np.sum(W @ np.square(a - Φ @ self.θ)) / Z)

                self.writer.add_scalar('rl/reward', torch.tensor(np.sum(rewards)), self.epoch)
                self.writer.add_scalar('rl/η', torch.tensor(self.η), self.epoch)
                self.writer.add_scalar('rl/KL', torch.tensor(self.kl), self.epoch)
                self.writer.add_scalar('rl/σ', torch.tensor(self.σ), self.epoch)

                np.savez(f"./out/models/{self.name}.npz", θ=self.θ, α=self.α, η=self.η, σ=self.σ,
                         fourier_features=self.fourier_features, epoch=self.epoch)

            print(f"{self.epoch:4} rewards {np.sum(rewards):13.6f} | KL {self.kl:8.6f} | σ {self.σ}")
