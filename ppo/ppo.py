import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import gym
import quanser_robots

class PolicyNetwork(nn.Module):

    def __init__(self, env):
        super(PolicyNetwork, self).__init__()
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.action_scaling = torch.tensor(self.action_space.high, dtype=torch.float32)

        self.h1 = nn.Linear(self.observation_space.shape[0], 100)
        self.action_mu  = nn.Linear(100, 1)
        self.action_sig = nn.Linear(100, 1)

    def forward(self, state):
        x = F.relu(self.h1(state))
        μ = self.action_scaling * torch.tanh(self.action_mu(x))
        σ = F.softplus(self.action_sig(x)) + 1e-5 # make sure we never predict zero variance.
        return μ, σ

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            μ, σ = self.forward(state)
        a_dist = Normal(μ, σ)
        action = a_dist.sample()
        log_prob = a_dist.log_prob(action)
        return action.item(), log_prob.item()


class ValueNetwork(nn.Module):

    def __init__(self, env):
        super(ValueNetwork, self).__init__()
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.h1 = nn.Linear(self.observation_space.shape[0], 100)
        self.value = nn.Linear(100, 1)

    def forward(self, state):
        #x = torch.from_numpy(state).float().unsqueeze(0)
        x = F.relu(self.h1(state))
        value = self.value(x)
        return value

    def get_value(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        value = self.forward(state)
        return value.item()


parser = argparse.ArgumentParser(description='Solve the different gym envs with PPO')
parser.add_argument('--eval', action='store_true', help='toggles evaluation mode')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--resume', action='store_true', help='resume training on an existing model by loading the last checkpoint')
args = parser.parse_args()

torch.set_printoptions(threshold=5000)

training = not args.eval



env = gym.make('Pendulum-v0')
# env = gym.make('DoublePendulum-v0') # DoubleCartPole
# env = gym.make('Qube-v0')  # FurutaPend
# env = gym.make('BallBalancerSim-v0')  # BallBalancer
print('Env:', env)
print('Reward range', env.reward_range)
print('Observation space:', env.observation_space)
print('  ', env.observation_space.high)
print('  ', env.observation_space.low)
print('Action space:', env.action_space)
print('  ', env.action_space.high)
print('  ', env.action_space.low)

# torch.manual_seed(0)
# env.seed(0)

model   = PolicyNetwork(env).float()
model_v = ValueNetwork(env).float()
optimizer   = optim.Adam(model.parameters(), lr=1e-4)
optimizer_v = optim.Adam(model_v.parameters(), lr=3e-4)
epoch = -1

if args.eval or args.resume:
    checkpoint = torch.load('out/ppo_checkpoint.pt')
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


while epoch < 100000:
    epoch += 1
    state = env.reset()
    states, next_states, actions, log_probs, rewards, values, dones = [],[],[],[],[],[],[]

    traj_length = 0
    traj_lengths = []

    # Sample trajectories
    n_steps = 1000
    for step in range(n_steps):

        action, log_prob = model.select_action(state)
        state_, reward, done, _ = env.step(np.array([action]))

        # since rendering is expensive only render sometimes
        if args.render:
            env.render()

        states.append(state)
        next_states.append(state_)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        # values.append(value)

        dones.append(done)

        traj_length += 1
        state = state_

        if done:
            state = env.reset()
            traj_lengths.append(traj_length)
            traj_length = 0

    traj_lengths.append(traj_length)

    # print()
    # print('states', states)
    # print('next_states', next_states)
    # print('actions', actions)
    # print('log_probs', log_probs)
    # print('rewards', rewards)
    # print('values', values)
    # print('dones', dones)
    # print()

    # calculate returns
    #################
    # Update Policy #
    #################

    γ = 0.9
    num_ppo_epochs = 10
    batch_size = 128

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32).view(-1,1)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    old_log_probs = torch.tensor(log_probs, dtype=torch.float32).view(-1,1)
    rewards = torch.tensor(rewards, dtype=torch.float32).view(-1,1)
    normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

    # REVIEW: use 1-step GAE vs. discounted rewards for return calculation.

    # R = model_v.get_value(state)
    # returns = []
    # for t in reversed(range(n_steps)):
    #     R = normalized_rewards[t] + (1-dones[t]) * γ * R
    #     returns.insert(0,R)
    # returns = torch.tensor(returns, dtype=torch.float32).view(-1,1)

    with torch.no_grad():
        returns = normalized_rewards + γ * model_v(next_states)
        # returns = rewards + γ * model_v(next_states)
        # returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        advs = returns - model_v(states)

    policy_losses, value_losses = [], []

    if training:
        for _ in range(num_ppo_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(n_steps)), batch_size, False):

                # Construct the mini batch
                mb_states = states[index]
                mb_actions = actions[index]
                mb_old_log_probs = old_log_probs[index]
                mb_returns = returns[index]
                mb_advs = advs[index]

                μ, σ = model(mb_states)
                a_dist = Normal(μ, σ)
                mb_log_probs = a_dist.log_prob(mb_actions)
                ratio = torch.exp(mb_log_probs - mb_old_log_probs)

                optimizer.zero_grad()
                surr1 = ratio * mb_advs
                surr2 = torch.clamp(ratio, 0.8, 1.2) * mb_advs
                policy_loss = -torch.min(surr1, surr2).mean()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                optimizer_v.zero_grad()
                value_loss = F.smooth_l1_loss(model_v(mb_states), mb_returns)
                value_loss.backward()
                nn.utils.clip_grad_norm_(model_v.parameters(), 0.5)
                optimizer_v.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())

        model_states = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_v_state_dict': model_v.state_dict(),
            'optimizer_v_state_dict': optimizer_v.state_dict(),
        }
        torch.save(model_states, './out/ppo_checkpoint.pt')

    if epoch%10==0:
        print(f"{epoch:4} rewards {rewards.sum().item():10.2f} | policy {np.mean(policy_losses):12.3f} | value {np.mean(value_losses):12.3f}")
