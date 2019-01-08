import sys

import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal, MultivariateNormal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

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

parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
parser.add_argument('--n_steps', type=int, default=3000,
                    help='number of agent steps when collecting trajectories for one epoch')

parser.add_argument('--p_lr', type=float, default=1e-4, help='initial learning rate policy network')
parser.add_argument('--v_lr', type=float, default=1e-4, help='initial learning rate value network')

parser.add_argument('--ppo_batch_size', type=int, default=64, help='PPO mini batch size')
parser.add_argument('--n_ppo_epochs', type=int, default=10, help='number of epochs of PPO optimization')

args = parser.parse_args()
writer = SummaryWriter(log_dir=f"./out/summary/{args.experiment_id}")

torch.set_printoptions(threshold=5000)

training = not args.eval


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
        σ = F.softplus(self.action_sig(x)) + 1e-12  # make sure we never predict zero variance.
        return μ, σ

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            μ, σ = self.forward(state)
            σ = σ if training else σ*0.0+1e-12
            Σ = torch.stack([σ_ * torch.eye(self.num_actions) for σ_ in σ])
        # print('μ', μ, 'σ', σ, 'Σ', Σ)
        # print('Σ', Σ)
        a_dist = MultivariateNormal(μ, Σ)
        action = a_dist.sample().squeeze(0)
        log_prob = a_dist.log_prob(action)
        # print('action', action, 'log_p', log_prob)
        return action.numpy(), log_prob.numpy()


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


class PPONetwork(nn.Module):

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
        self.value = nn.Linear(n_neurons, 1)

    def forward(self, state):
        x = torch.tanh(self.h1(state))
        x = torch.tanh(self.h2(x))
        μ = self.action_scaling * torch.tanh(self.action_mu(x))
        σ = F.softplus(self.action_sig(x)) + 1e-12  # make sure we never predict zero variance.
        value = self.value(x)
        return μ, σ, value

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            μ, σ, _ = self.forward(state)
            Σ = torch.stack([σ_ * torch.eye(self.num_actions) for σ_ in σ])
        # print('μ', μ, 'σ', σ, 'Σ', Σ)
        # print('Σ', Σ)
        a_dist = MultivariateNormal(μ, Σ)
        action = a_dist.sample().squeeze(0)
        log_prob = a_dist.log_prob(action)
        # print('action', action, 'log_p', log_prob)
        return action.numpy(), log_prob.numpy()

    def get_value(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        _, _, value = self.forward(state)
        return value.item()

environments = {
    'balancer': 'BallBalancerSim-v0',
    'double_pendulum': 'DoublePendulum-v0',
    'double_pendulumRR': 'DoublePendulumRR-v0',
    'furuta': 'Qube-v0',
    'furutaRR': 'QubeRR-v0',
    'pendulum': 'Pendulum-v0',
    'swingup': 'CartpoleSwingShort-v0',
    'swingupRR': 'CartpoleSwingRR-v0',
    'stab': 'CartpoleStabShort-v0',
    'stabRR': 'CartpoleStabRR-v0',
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

print('p_lr', args.p_lr, 'v_lr', args.v_lr)

model = PolicyNetwork(env).float()
model_v = ValueNetwork(env).float()
optimizer = optim.Adam(model.parameters(), lr=args.p_lr)  # 3e-3
optimizer_v = optim.Adam(model_v.parameters(), lr=args.v_lr)  # 6e-3

epoch = 0
n_steps = args.n_steps
γ = args.gamma

if args.eval or args.resume:
    checkpoint = torch.load(f"./out/models/{args.experiment_id}.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    model_v.load_state_dict(checkpoint['model_v_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    optimizer_v.load_state_dict(checkpoint['optimizer_v_state_dict'])
    epoch = checkpoint['epoch']
    print(f"-> LOADED MODEL at epoch {epoch}")

if training:
    model.train()
    model_v.train()
else:
    model.eval()
    model_v.eval()

while epoch < 10000:
    epoch += 1
    state = env.reset()
    states, next_states, actions, log_probs, rewards, values, dones = [], [], [], [], [], [], []

    # Sample trajectories
    for step in range(n_steps):

        action, log_prob = model.select_action(state)
        state_, reward, done, _ = env.step(action)

        if args.render:
            env.render()

        states.append(state)
        next_states.append(state_)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        dones.append(done)

        if done:
            state = env.reset()
        else:
            state = state_
        print(f"step {step}", end="\r")

    # print()
    # print('states', states)
    # print('next_states', next_states)
    # print('actions', actions)
    # print('log_probs', log_probs)
    # print('rewards', rewards)
    # print('values', values)
    # print('dones', dones)
    # print()

    #################
    # Update Policy #
    #################

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32).view(-1, env.action_space.shape[0])
    next_states = torch.tensor(next_states, dtype=torch.float32)
    old_log_probs = torch.tensor(log_probs, dtype=torch.float32).view(-1, 1)
    rewards = torch.tensor(rewards, dtype=torch.float32).view(-1, 1)
    dones = torch.tensor(dones, dtype=torch.float32).view(-1, 1)
    normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
    mean_reward = rewards.sum() / (1 - dones).sum()
    policy_losses, value_losses = [], []

    if training:
        # --------------------------------
        #   Return calculation
        # --------------------------------

        # - Monte Carlo --------------
        R = model_v.get_value(state)
        # returns = []
        returns = torch.zeros(n_steps)
        for t in reversed(range(n_steps)):
            # R = normalized_rewards[t] + (1-dones[t]) * γ * R
            R = rewards[t] + (1-dones[t]) * γ * R
            # returns.insert(0, R)
            returns[t] = R
        returns = returns.view(-1, 1)
        # returns = torch.tensor(returns, dtype=torch.float32).view(-1,1)

        # - Temporal Difference ------
        # with torch.no_grad():
        #     # returns = normalized_rewards + γ * model_v(next_states)
        #     returns = rewards + γ * model_v(next_states)

        # Update Value function
        for _ in range(args.n_ppo_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(n_steps)), args.ppo_batch_size, False):
                optimizer_v.zero_grad()
                mb_states = states[index]
                mb_returns = returns[index]
                value_loss = F.smooth_l1_loss(model_v(mb_states), mb_returns)
                value_loss.backward()
                nn.utils.clip_grad_norm_(model_v.parameters(), 0.5)
                optimizer_v.step()
                value_losses.append(value_loss.item())

        # --------------------------------
        #   Advantage Calculation
        # --------------------------------

        # - GAE ----------------------
        # with torch.no_grad():
        #     λ = 0.99
        #     A_GAE = 0.0
        #     advs = torch.zeros(n_steps)
        #     δ = rewards + γ * model_v(next_states) * (1 - dones) - model_v(states)
        #     # δ = rewards + γ * model_v(next_states) - model_v(states)
        #     for t in reversed(range(n_steps)):
        #         A_GAE = δ[t] + λ * γ * A_GAE * (1-dones[t])
        #         advs[t] = A_GAE
        #     # advs = torch.tensor(advs).view(-1, 1)

        # - Temporal Difference ------
        with torch.no_grad():
            advs = returns - model_v(states)
            advs = (advs - advs.mean()) / advs.std()

        for _ in range(args.n_ppo_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(n_steps)), args.ppo_batch_size, False):
                # Construct the mini batch
                mb_states = states[index]
                mb_actions = actions[index]
                mb_old_log_probs = old_log_probs[index]
                mb_advs = advs[index]

                μ, σ = model(mb_states)
                Σ = torch.stack([σ_ * torch.eye(env.action_space.shape[0]) for σ_ in σ])
                a_dist = MultivariateNormal(μ, Σ)
                mb_log_probs = a_dist.log_prob(mb_actions).unsqueeze(-1)
                ratio = torch.exp(mb_log_probs - mb_old_log_probs)

                optimizer.zero_grad()
                surr1 = ratio * mb_advs
                surr2 = torch.clamp(ratio, 0.95, 1.05) * mb_advs
                policy_loss = -torch.min(surr1, surr2).mean()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                policy_losses.append(policy_loss.item())

        model_states = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_v_state_dict': model_v.state_dict(),
            'optimizer_v_state_dict': optimizer_v.state_dict(),
        }
        torch.save(model_states, f"./out/models/{args.experiment_id}.pt")

        writer.add_scalar('rl/reward', mean_reward, epoch)

    # if epoch%10==0:
    print(f"{epoch:4} rewards {mean_reward.item():10.6f} | policy {np.mean(policy_losses):12.3f} |\
     value {np.mean(value_losses):12.3f}")
