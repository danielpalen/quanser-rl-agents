import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal

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
        x = torch.from_numpy(state).float().unsqueeze(0)
        x = torch.tanh(self.h1(x))
        μ = self.action_scaling * torch.tanh(self.action_mu(x))
        σ = F.softplus(self.action_sig(x)) + 0.01 # make sure we never predict zero variance.
        μ = μ.view(-1)
        σ = σ.view(-1)
        return μ, σ


class ValueNetwork(nn.Module):

    def __init__(self, env):
        super(ValueNetwork, self).__init__()
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.h1 = nn.Linear(self.observation_space.shape[0], 100)
        self.value = nn.Linear(100, 1)

    def forward(self, state):
        x = torch.from_numpy(state).float().unsqueeze(0)
        x = torch.tanh(self.h1(x))
        value = self.value(x).view(-1)
        return value


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


model   = PolicyNetwork(env).float()
model_v = ValueNetwork(env).float()
optimizer   = optim.Adam(model.parameters(), lr=1e-4)
optimizer_v = optim.Adam(model_v.parameters(), lr=1e-4)
epoch = 0

if args.eval or args.resume:
    checkpoint = torch.load('out/checkpoint.pt')
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
    n_steps = 2000
    for step in range(n_steps):

        (μ, σ), value = model(state), model_v(state)

        a_dist = Normal(μ, σ)
        action = a_dist.sample()
        log_prob = a_dist.log_prob(action)
        a = action.detach().numpy()
        state_, reward, done, _ = env.step(a)

        # since rendering is expensive only render sometimes
        if args.render and (epoch%40==0 and epoch>0 or args.eval):
            env.render()

        states.append(state)
        next_states.append(state_)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        values.append(value)
        dones.append(done)

        traj_length += 1
        state = state_

        if done:
            state = env.reset()
            traj_lengths.append(traj_length)
            traj_length = 0

    traj_lengths.append(traj_length)

    # Update Policy
    γ = 0.95
    returns = []
    R = model_v(state).item()

    for t in reversed(range(n_steps)):
        R = rewards[t] + (1-dones[t]) * γ * R
        returns.insert(0,R)

    dones = torch.tensor(dones, dtype=torch.float32).view(-1,1)
    rewards = torch.tensor(rewards, dtype=torch.float32).view(-1,1)
    returns = torch.tensor(returns, dtype=torch.float32).view(-1,1)
    advs = returns - torch.tensor(values, dtype=torch.float32).view(-1,1)

    policy_losses, value_losses = [], []

    for log_prob, adv, value, r in zip(log_probs, advs, values, returns):
        policy_losses.append( -log_prob * adv.item() )
        value_losses.append( (value - r)**2 )

    policy_loss = torch.stack(policy_losses).sum()
    value_loss = torch.stack(value_losses).sum()
    loss = policy_loss + value_loss

    if training:
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        optimizer_v.zero_grad()
        value_loss.backward()
        optimizer_v.step()

        # checkpointing
        if epoch%10==0:
            model_states = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'model_v_state_dict': model_v.state_dict(),
                'optimizer_v_state_dict': optimizer_v.state_dict(),
                'loss': loss,
            }
            torch.save(model_states, './out/checkpoint.pt')
            print('SAVED checkpoint')

    print(f"epoch {epoch:3} -> steps {np.mean(traj_lengths):4.0f}  | rewards {torch.sum(rewards)/len(traj_lengths):8.4f} {torch.sum(rewards):8.0f} |   loss {loss:12.2f}  |   policy {policy_loss:12.2f}  |   value {value_loss:8.2f}")
