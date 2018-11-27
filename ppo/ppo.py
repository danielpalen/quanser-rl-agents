import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal

import gym
import quanser_robots


parser = argparse.ArgumentParser(description='Solve the different gym envs with PPO')
parser.add_argument('--render', action='store_true', help='render the environment')
args = parser.parse_args()

torch.set_printoptions(threshold=5000)


class PolicyNetwork(nn.Module):
    """
    Proximal Policy Optimization algorithm.
    Paper: https://arxiv.org/abs/1707.06347
    """

    def __init__(self, env):
        super(PolicyNetwork, self).__init__()

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.h1 = nn.Linear(self.observation_space.shape[0], 100)
        #self.h2 = nn.Linear(100, 100)

        self.action_mu  = nn.Linear(100, 1)
        self.action_sig = nn.Linear(100, 1)
        #self.value = nn.Linear(100, 1)

    def forward(self, state):
        # print('state', state.shape)
        x = torch.from_numpy(state).float().unsqueeze(0)
        # print('x', x)
        x = torch.tanh(self.h1(x))
        # x = F.relu(self.h2(x))

        # tanh gives values between [-1,1]. So we scale μ according to the
        # environment's specifications.
        # print('tensor', torch.tensor(self.action_space.high, dtype=torch.float32))
        μ = torch.tensor(self.action_space.high, dtype=torch.float32) * torch.tanh(self.action_mu(x))
        σ = F.softplus(self.action_sig(x)) + 0.01 # make sure we never predict zero variance.
        #value = self.value(x)
        # print('μ', μ)
        # print('σ', σ)
        # print()
        return μ.view(-1), σ.view(-1) #, value


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
        # print('value', value)
        return value


def train():

    env = gym.make('Pendulum-v0')
    # env = gym.make('DoublePendulum-v0') # DoubleCartPolea
    # env = gym.make('Qube-v0')  # FurutaPend
    # env = gym.make('BallBalancerSim-v0')  # BallBalancer

    model   = PolicyNetwork(env).float()
    model_v = ValueNetwork(env).float()

    optimizer   = optim.Adam(model.parameters(), lr=1e-3)
    optimizer_v = optim.Adam(model_v.parameters(), lr=1e-3)

    print('Env:', env)
    print('Reward range', env.reward_range)
    print('Observation space:', env.observation_space)
    print('  ', env.observation_space.high)
    print('  ', env.observation_space.low)
    print('Action space:', env.action_space)
    print('  ', env.action_space.high)
    print('  ', env.action_space.low)

    for epoch in range(100000):
        state = env.reset()
        states, next_states, actions, log_probs, rewards, values, dones = [],[],[],[],[],[],[]

        traj_length = 0
        traj_lengths = []

        # Sample trajectories
        n_steps = 2000
        for step in range(n_steps):

            #print(f"  step {step+1:4}/{n_steps}", end="\r")

            # Sample from network
            (μ, σ), value = model(state), model_v(state)
            # print('(μ, σ), value', (μ, σ), value)
            # print()

            a_dist = Normal(μ, σ)
            action = a_dist.sample()
            log_prob = a_dist.log_prob(action)
            a = action.detach().numpy()
            # print('a', a)
            state_, reward, done, _ = env.step(a)

            # print('state_', state_)

            # since rendering is expensive only render sometimes
            if args.render and epoch%20==0 and epoch > 0 and step < 500:
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
        # R = model(state)[2].detach().item()
        R = model_v(state).item()
        for t in reversed(range(n_steps)):
            R = rewards[t] + (1-dones[t]) * γ * R
            returns.insert(0,R)

        dones = torch.tensor(dones, dtype=torch.float32).view(-1,1)
        values_ = torch.tensor(values, dtype=torch.float32).view(-1,1)
        rewards = torch.tensor(rewards, dtype=torch.float32).view(-1,1)
        returns = torch.tensor(returns, dtype=torch.float32).view(-1,1)
        advantages = returns - values_

        policy_losses = []
        value_losses = []
        for log_prob, adv, value, r in zip(log_probs, advantages, values, returns):
            policy_losses.append(-log_prob * adv.item())
            value_losses.append( (value - r)**2 )

        policy_loss = torch.stack(policy_losses).sum()
        value_loss = torch.stack(value_losses).sum()
        loss = policy_loss + value_loss

        #############################
        # Single network optimization
        #############################

        # optimizer.zero_grad()
        #
        # # for f in model.parameters():
        # #     print()
        # #     print(f'{f.data.size()} grad is')
        # #     print(f.grad)
        #
        # loss.backward()
        #
        # # for f in model.parameters():
        # #     print()
        # #     print(f'{f.data.size()} grad is')
        # #     print(f.grad)
        #
        # optimizer.step()


        ###########################
        # Two networks optimization
        ###########################

        optimizer.zero_grad()
        # for f in model.parameters():
        #     print()
        #     print(f'{f.data.size()} grad is')
        #     print(f.grad)
        policy_loss.backward()
        # for f in model.parameters():
        #     print()
        #     print(f'{f.data.size()} grad is')
        #     print(f.grad)
        optimizer.step()

        optimizer_v.zero_grad()
        # print('Before Backward')
        # for f in model_v.parameters():
        #     print()
        #     print(f'{f.data.size()} grad is')
        #     print(f.grad)
        value_loss.backward()
        # print('\n\nAfter Backward')
        # for f in model_v.parameters():
        #     print()
        #     print(f'{f.data.size()} grad is')
        #     print(f.grad)
        optimizer_v.step()




        print(f"epoch {epoch:3} -> steps {np.mean(traj_lengths):4.0f}  | rewards {torch.sum(rewards)/len(traj_lengths):8.4f} {torch.sum(rewards):8.0f} |   loss {loss:12.2f}  |   policy {policy_loss:12.2f}  |   value {value_loss:8.2f}")


if __name__ == '__main__':
    train()
