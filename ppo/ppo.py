import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from tensorboardX import SummaryWriter


class PolicyNetwork(nn.Module):

    def __init__(self, env):
        super(PolicyNetwork, self).__init__()
        self.observation_space = env.observation_space
        self.num_actions = env.action_space.shape[0]
        self.action_scaling = torch.tensor(env.action_space.high, dtype=torch.float32)

        n_neurons = 64

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

    def select_action(self, state, training=True):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            μ, σ = self.forward(state)
            σ = σ if training else torch.zeros_like(σ)+1e-12
            Σ = torch.stack([σ_ * torch.eye(self.num_actions) for σ_ in σ])  # TODO: optimize?
        a_dist = MultivariateNormal(μ, Σ)
        action = a_dist.sample().squeeze(0)
        log_prob = a_dist.log_prob(action)
        return action.numpy(), log_prob.numpy(), Σ


class ValueNetwork(nn.Module):

    def __init__(self, env):
        super(ValueNetwork, self).__init__()
        self.observation_space = env.observation_space

        n_neurons = 64

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


class PPO:
    """
    Proximal Policy Optimization based on https://arxiv.org/abs/1707.06347
    """

    def __init__(self, *, name, env, n_epochs=100, n_steps=3000, gamma=0.9, p_lr=7e-4, v_lr=7e-4, n_mb_epochs=5,
                 mb_size=32, clip=0.2, gae=False, lam=0.99, render=False, resume=False, eval=False, seed=None,
                 summary_path=None, checkpoint_path=None, **kwargs):

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
        self.n_mb_epochs = n_mb_epochs
        self.mb_size = mb_size

        self.policy = PolicyNetwork(env).float()
        self.V = ValueNetwork(env).float()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=p_lr)
        self.optimizer_v = optim.Adam(self.V.parameters(), lr=v_lr)

        self.epoch = 0

        if self.eval or self.resume:
            checkpoint = torch.load(self.checkpoint_path)
            self.policy.load_state_dict(checkpoint['model_state_dict'])
            self.V.load_state_dict(checkpoint['model_v_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.optimizer_v.load_state_dict(checkpoint['optimizer_v_state_dict'])
            self.epoch = checkpoint['epoch']
            print(f"-> LOADED MODEL at epoch {self.epoch}")

        if self.training:
            self.policy.train()
            self.V.train()
        else:
            self.policy.eval()
            self.V.eval()

    def train(self):
        while self.epoch < self.n_epochs or self.eval:
            self.epoch += 1
            state = self.env.reset()
            states, next_states, actions, log_probs, covs, rewards, values, dones = [], [], [], [], [], [], [], []

            # Sample trajectories
            for step in range(self.n_steps):

                action, log_prob, Σ = self.policy.select_action(state, training=self.training)
                state_, reward, done, _ = self.env.step(action)

                if self.render:
                    self.env.render()

                states.append(state)
                next_states.append(state_)
                actions.append(action)
                log_probs.append(log_prob)
                covs.append(Σ.numpy())
                rewards.append(reward)
                dones.append(done)

                if done:
                    state = self.env.reset()
                else:
                    state = state_
                print(f"step {step}", end="\r")

            #################
            # Update Policy #
            #################

            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.float32).view(-1, self.env.action_space.shape[0])
            next_states = torch.tensor(next_states, dtype=torch.float32)
            old_log_probs = torch.tensor(log_probs, dtype=torch.float32).view(-1, 1)
            rewards = torch.tensor(rewards, dtype=torch.float32).view(-1, 1)
            dones = torch.tensor(dones, dtype=torch.float32).view(-1, 1)
            # normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            mean_reward = rewards.sum()  # / (dones.sum() + 1)
            policy_losses, value_losses = [], []

            if self.training:
                ##############################
                # Return calculation         #
                # ############################

                # - Monte Carlo --------------
                R = self.V.get_value(state)
                returns = torch.zeros(self.n_steps)
                for t in reversed(range(self.n_steps)):
                    # R = normalized_rewards[t] + (1-dones[t]) * γ * R
                    R = rewards[t] + (1 - dones[t]) * self.γ * R
                    # returns.insert(0, R)
                    returns[t] = R
                returns = returns.view(-1, 1)
                # returns = torch.tensor(returns, dtype=torch.float32).view(-1,1)

                # - Temporal Difference ------
                # with torch.no_grad():
                #     # returns = normalized_rewards + γ * V(next_states)
                #     returns = rewards + γ * V(next_states)

                # Update Value function
                # for _ in range(args.n_ppo_epochs):
                #     for index in BatchSampler(SubsetRandomSampler(range(n_steps)), args.ppo_batch_size, False):
                #         # Construct the mini batch
                #         mb_states = states[index]
                #         mb_returns = returns[index]
                #
                #         optimizer_v.zero_grad()
                #         value_loss = F.smooth_l1_loss(V(mb_states), mb_returns)
                #         value_loss.backward()
                #         nn.utils.clip_grad_norm_(V.parameters(), 0.5)
                #         optimizer_v.step()
                #
                #         value_losses.append(value_loss.item())

                ##############################
                # Advantage Calculation      #
                ##############################
                with torch.no_grad():
                    if self.gae:
                        # - GAE ----------------------
                        A_GAE = 0.0
                        advs = torch.zeros(self.n_steps)
                        δ = rewards + self.γ * self.V(next_states) * (1 - dones) - self.V(states)
                        # δ = rewards + γ * V(next_states) - V(states)
                        for t in reversed(range(self.n_steps)):
                            A_GAE = δ[t] + self.λ * self.γ * A_GAE * (1-dones[t])
                            advs[t] = A_GAE
                        advs = advs.view(-1, 1)
                    else:
                        # - Temporal Difference ------
                        advs = returns - self.V(states)

                for _ in range(self.n_mb_epochs):
                    for index in BatchSampler(SubsetRandomSampler(range(self.n_steps)), self.mb_size, False):
                        # Construct the mini batch
                        mb_states = states[index]
                        mb_actions = actions[index]
                        mb_old_log_probs = old_log_probs[index]
                        mb_returns = returns[index]
                        mb_advs = advs[index]

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

                model_states = {
                    'epoch': self.epoch,
                    'model_state_dict': self.policy.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'model_v_state_dict': self.V.state_dict(),
                    'optimizer_v_state_dict': self.optimizer_v.state_dict(),
                }
                torch.save(model_states, self.checkpoint_path)

                self.writer.add_scalar('rl/reward', mean_reward, self.epoch)
                self.writer.add_scalar('rl/entropy', normal_entropy(covs), self.epoch)

            print(f"{self.epoch:4} rewards {mean_reward.item():13.3f} | policy {torch.stack(policy_losses).mean():12.3f}\
             | value {torch.stack(value_losses).mean():12.3f} | σ {np.mean(covs)} | entropy {normal_entropy(covs)}")


def normal_entropy(covariances):
    """Calculates the mean entropy of gaussian polies given a list of covariance matrices Σ."""
    return torch.mean(torch.from_numpy(0.5 * np.log(2*np.e*np.pi*np.linalg.det(covariances))))
