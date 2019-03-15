import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from tensorboardX import SummaryWriter

from common.common import save_tb_scalars


class PPO:

    def __init__(self, *, name, env, n_epochs=100, n_steps=3000, gamma=0.9, p_lr=7e-4, v_lr=7e-4, n_mb_epochs=5,
                 mb_size=64, clip=0.2, td=False, gae=False, lam=0.99, render=False, resume=False, eval=False, seed=None,
                 summary_path=None, checkpoint_path=None, **kwargs):
        """
        Proximal Policy Optimization based on https://arxiv.org/abs/1707.06347

        :param name: experiment name for checkpointing
        :param env: instance of OpenAI gym environment
        :param n_epochs: number of training epochs
        :param n_steps: number of environment steps per training epoch
        :param gamma: discount factor for return calculation
        :param p_lr: learning rate for the policy network
        :param v_lr: learning rate for the value network
        :param n_mb_epochs: number of mini batch update epochs
        :param mb_size: size of the mini batches
        :param clip: clipping factor used in the PPO objective function
        :param td: if True returns will be calculated using Temporal Difference otherwise Monte Carlo estimates
                   will be used.
        :param gae: if True the advantage will be calculated via Generalized Advantage Estimation otherwise
                    (return-value) will be used.
        :param lam: λ hyperparameter used by the Generalized Advantage Estimation algorithm,
        :param render: renders the environment.
        :param resume: loads the last checkpoint to continue to train.
        :param eval: loads the last checkpoint to perform evaluation of the deterministic policy afterwards..
        :param seed: optional seed.
        :param summary_path: path at which tensorboard summary files are saved.
        :param checkpoint_path: path at which model checkpoints are saved and loaded.
        :param kwargs: Helper to catch unused arguments supplied by the argument parser in run.py
        """

        if seed is not None:
            torch.manual_seed(seed)

        self.name = name
        self.env = env
        self.n_epochs = n_epochs
        self.n_steps = n_steps
        self.render = render
        self.resume = resume
        self.eval = eval
        self.training = not eval
        self.summary_path = summary_path
        self.checkpoint_path = os.path.join(checkpoint_path, self.name + '.pt')
        self.writer = SummaryWriter(log_dir=self.summary_path)

        self.clip = clip
        self.γ = gamma
        self.λ = lam
        self.gae = gae
        self.td = td
        self.n_mb_epochs = n_mb_epochs
        self.mb_size = mb_size

        self.policy = PolicyNetwork(env).float()
        self.V = ValueNetwork(env).float()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=p_lr)
        self.optimizer_v = optim.Adam(self.V.parameters(), lr=v_lr)

        self.epoch = 0

        if self.eval or self.resume:
            self.load_model()


    def train(self):
        """Trains the PPO agent."""

        while self.epoch < self.n_epochs:
            self.epoch += 1
            self.policy.train()
            self.V.train()
            state = self.env.reset()
            states, next_states, actions, log_probs, covs, rewards, values, dones = [], [], [], [], [], [], [], []

            for step in range(self.n_steps):
                action, log_prob, Σ = self.policy.select_action(state)
                next_state, reward, done, _ = self.env.step(action)

                if self.render:
                    self.env.render()

                states.append(state)
                next_states.append(next_state)
                actions.append(action)
                log_probs.append(log_prob)
                covs.append(Σ.numpy())
                rewards.append(reward)
                dones.append(done)

                if done:
                    state = self.env.reset()
                else:
                    state = next_state
                print(f"step {step}", end="\r")

            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.float32).view(-1, self.env.action_space.shape[0])
            next_states = torch.tensor(next_states, dtype=torch.float32)
            old_log_probs = torch.tensor(log_probs, dtype=torch.float32).view(-1, 1)
            rewards = torch.tensor(rewards, dtype=torch.float32).view(-1, 1)
            dones = torch.tensor(dones, dtype=torch.float32).view(-1, 1)
            policy_losses, value_losses = [], []

            # Return calculation
            if self.td:
                # Temporal Difference
                with torch.no_grad():
                    returns = rewards + self.γ * self.V(next_states)
            else:
                # Monte Carlo
                R = self.V.get_value(state)
                returns = torch.zeros(self.n_steps)
                for t in reversed(range(self.n_steps)):
                    R = rewards[t] + (1 - dones[t]) * self.γ * R
                    returns[t] = R
                returns = returns.view(-1, 1)

            # Advantage Calculation
            with torch.no_grad():
                if self.gae:
                    # Generalized Advantage Estimation
                    A_GAE = 0.0
                    advs = torch.zeros(self.n_steps)
                    δ = rewards + self.γ * self.V(next_states) * (1 - dones) - self.V(states)
                    for t in reversed(range(self.n_steps)):
                        A_GAE = δ[t] + self.λ * self.γ * A_GAE * (1-dones[t])
                        advs[t] = A_GAE
                    advs = advs.view(-1, 1)
                else:
                    # Temporal Difference
                    advs = returns - self.V(states)

            # Mini batch policy and value function updates
            for _ in range(self.n_mb_epochs):
                for indices in BatchSampler(SubsetRandomSampler(range(self.n_steps)), self.mb_size, False):
                    # select samples for the mini batches
                    mb_states = states[indices]
                    mb_actions = actions[indices]
                    mb_old_log_probs = old_log_probs[indices]
                    mb_returns = returns[indices]
                    mb_advs = advs[indices]

                    μ, σ = self.policy(mb_states)
                    Σ = torch.stack([σ_ * torch.eye(self.env.action_space.shape[0]) for σ_ in σ])  # TODO: optimize?
                    a_dist = MultivariateNormal(μ, Σ)
                    mb_log_probs = a_dist.log_prob(mb_actions).unsqueeze(-1)
                    ratio = torch.exp(mb_log_probs - mb_old_log_probs)

                    self.optimizer.zero_grad()
                    surr1 = ratio * mb_advs
                    surr2 = torch.clamp(ratio, 1-self.clip, 1+self.clip) * mb_advs
                    policy_loss = -torch.min(surr1, surr2).mean()
                    policy_loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                    self.optimizer.step()

                    self.optimizer_v.zero_grad()
                    value_loss = (self.V(mb_states) - mb_returns).pow(2).mean()
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(self.V.parameters(), 0.5)
                    self.optimizer_v.step()

                    policy_losses.append(policy_loss)
                    value_losses.append(value_loss)
            self.save_model()

            if self.epoch%10==0:
                # Evaluate the deterministic policy
                n_eval_traj = 25
                _, mean_traj_reward = self.evaluate(n_eval_traj)
                entropy = normal_entropy(covs)
                save_tb_scalars(self.writer, self.epoch, reward=rewards.sum(), mean_traj_reward=mean_traj_reward,
                                entropy=entropy, policy_loss=torch.stack(policy_losses).mean(),
                                value_loss=torch.stack(value_losses).mean())
            else:
                entropy = normal_entropy(covs)
                save_tb_scalars(self.writer, self.epoch, reward=rewards.sum(), entropy=entropy,
                                policy_loss=torch.stack(policy_losses).mean(),
                                value_loss=torch.stack(value_losses).mean())

    def evaluate(self, n_trajectories, print_reward=False):
        """
        Evaluate the deterministic policy for N full trajectories.

        :param n_trajectories: number of trajectories to use for the evaluation.
        :return (cumulative_reward, mean_traj_reward):
        """
        self.policy.eval()
        self.V.eval()
        cumulative_reward = 0
        trajectory = 0
        traj_rewards = []
        traj_reward = 0
        step = 0
        state = self.env.reset()

        actions = []
        states = []

        print('Evaluating the deterministic policy...')

        while len(traj_rewards) < n_trajectories:
            step += 1
            action, _, _ = self.policy.select_action(state, deterministic=True)
            states.append(state)
            actions.append(action)
            next_state, reward, done, _ = self.env.step(action)
            cumulative_reward += reward
            traj_reward += reward
            if self.render:
                self.env.render()
            # state = next_state
            if done:
                # print(step)
                step = 0
                state = self.env.reset()
                trajectory += 1
                traj_rewards.append(traj_reward)
                traj_reward = 0
                if print_reward:
                    # print(traj_rewards)
                    print(len(traj_rewards), 'trajectories: total', cumulative_reward, 'mean', np.mean(traj_rewards), 'std', np.std(traj_rewards),
                          'max', np.max(traj_rewards))
                    # print('states', states)
                    # print('actions', actions)
                    # print()
                states = []
                actions = []
            else:
                state = next_state
        mean_traj_reward = cumulative_reward / n_trajectories
        if print_reward:
            print('FINAL: total', cumulative_reward, 'mean', np.mean(traj_rewards), 'std', np.std(traj_rewards), 'max', np.max(traj_rewards))
            print()

        return cumulative_reward, mean_traj_reward

    def load_model(self):
        """Loads a model checkpoint"""
        checkpoint = torch.load(self.checkpoint_path)
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.V.load_state_dict(checkpoint['model_v_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.optimizer_v.load_state_dict(checkpoint['optimizer_v_state_dict'])
        self.epoch = checkpoint['epoch']
        print(f"-> LOADED MODEL at epoch {self.epoch}")

    def save_model(self):
        """Saves model checkpoint"""
        model_states = {
            'epoch': self.epoch,
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_v_state_dict': self.V.state_dict(),
            'optimizer_v_state_dict': self.optimizer_v.state_dict(),
        }
        torch.save(model_states, self.checkpoint_path)


class PolicyNetwork(nn.Module):

    def __init__(self, env, n_neurons=64):
        """
        Policy network for PPO. This network predicts the parameters
        for a gaussian policy, i.e. mean and variance.

        :param env: gym environment needed to look at the observation space and action space size
        :param n_neurons: number of neurons in the hidden layers.
        """
        super(PolicyNetwork, self).__init__()
        self.observation_space = env.observation_space
        self.num_actions = env.action_space.shape[0]
        self.action_scaling = torch.tensor(env.action_space.high, dtype=torch.float32)
        self.h1 = nn.Linear(self.observation_space.shape[0], n_neurons)
        self.h2 = nn.Linear(n_neurons, n_neurons)
        self.action_mu = nn.Linear(n_neurons, self.num_actions)
        self.action_sig = nn.Linear(n_neurons, self.num_actions)

    def forward(self, state):
        x = torch.tanh(self.h1(state))
        x = torch.tanh(self.h2(x))
        μ = self.action_scaling * torch.tanh(self.action_mu(x))
        σ = 10 * F.softplus(self.action_sig(x)) + 1e-12  # make sure we never predict zero variance.
        return μ, σ

    def select_action(self, state, deterministic=False):
        """
        Samples an action from the policy network
        :param state: the environment state
        :param deterministic: whether the sampling should evaluate the deterministic or the stochastic policy.
                              For training we will use the stochastic one but for evaluation the deterministic one.
        :return: the action, it's log probability and the covariance matrix.
        """
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            μ, σ = self.forward(state)
            σ = σ if not deterministic else torch.zeros_like(σ)+1e-12
            Σ = torch.stack([σ_ * torch.eye(self.num_actions) for σ_ in σ])  # TODO: optimize?
        a_dist = MultivariateNormal(μ, Σ)
        action = a_dist.sample().squeeze(0)
        log_prob = a_dist.log_prob(action)
        return action.numpy(), log_prob.numpy(), Σ


class ValueNetwork(nn.Module):

    def __init__(self, env, n_neurons = 64):
        """
        Value Function Network

        :param env: gym environment needed to look at the observation space size
        :param n_neurons: number of neurons in the hidden layers.
        """
        super(ValueNetwork, self).__init__()
        self.observation_space = env.observation_space
        self.h1 = nn.Linear(self.observation_space.shape[0], n_neurons)
        self.h2 = nn.Linear(n_neurons, n_neurons)
        self.value = nn.Linear(n_neurons, 1)

    def forward(self, state):
        x = torch.tanh(self.h1(state))
        x = torch.tanh(self.h2(x))
        value = self.value(x)
        return value

    def get_value(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        value = self.forward(state)
        return value.item()


def normal_entropy(covariances):
    """Calculates the mean entropy of gaussian polies given a list of covariance matrices Σ."""
    return torch.mean(torch.from_numpy(0.5 * np.log(2*np.e*np.pi*np.linalg.det(covariances))))
