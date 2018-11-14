import argparse
import numpy as np
from collections import namedtuple

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

# Transition = namedtuple('Transition',
#                        ('state', 'action', 'next_state', 'reward'))


class PolicyNetwork(nn.Module):
    """
    Proximal Policy Optimization algorithm.
    Paper: https://arxiv.org/abs/1707.06347
    """

    def __init__(self, env):
        super(PolicyNetwork, self).__init__()

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.fc = nn.Linear(self.observation_space.shape[0], 100)
        # Predict 1 mu and 1 sigma for each action independently in the future.
        # for a in action_space.shape[0]):
        self.action_mu  = nn.Linear(100, 1)
        self.action_sig = nn.Linear(100, 1)
        self.value = nn.Linear(100, 1)

    def forward(self, state):
        x = torch.from_numpy(state).float().unsqueeze(0)
        x = F.relu(self.fc(x))
        μ = 24.0 * torch.tanh(self.action_mu(x))
        σ = F.softplus(self.action_sig(x)) + 0.001
        value = self.value(x)
        return μ, σ, value


def train():

    # env = gym.make('Pendulum-v0')
    env = gym.make('DoublePendulum-v0') # DoubleCartPolea
    # env = gym.make('Qube-v0')  # FurutaPend
    # env = gym.make('BallBalancerSim-v0g')  # BallBalancer

    model = PolicyNetwork(env).float()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print('Env:', env)
    print('Reward range', env.reward_range)
    print('Observation space:', env.observation_space)
    print('  ', env.observation_space.high)
    print('  ', env.observation_space.low)
    print('Action space:', env.action_space)
    print('  ', env.action_space.high)
    print('  ', env.action_space.low)


    for epoch in range(10000):
        state = env.reset()
        states, actions, log_probs, rewards, values, dones = [],[],[],[],[],[]

        traj_length = 0
        traj_lengths = []

        # Sample trajectory
        n_steps = 2000
        for step in range(n_steps):

            print(f"  step {step+1:4}/{n_steps}", end="\r")

            # Sample from network
            μ, σ, value = model(state)
            a_dist = Normal(μ, σ)
            action = a_dist.sample()
            log_prob = a_dist.log_prob(action)
            action.clamp(
                float(env.action_space.low[0]),
                float(env.action_space.high[0])
            )

            # print(f"action {action:5.2f} | mu {mu.data} | sig {sig.data} | log_prob {log_prob}")
            # print('log_prob', log_prob)
            # print('value', value)
            # print('obs', state)
            # print()

            state_,reward,done,_ = env.step(action.numpy())

            # since rendering is expensive only render sometimes
            if args.render and epoch%20==0 and epoch > 0 and step < 500:
                env.render()

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            dones.append(done)

            if done:
                state = env.reset()
                traj_lengths.append(traj_length)
                traj_length = 0
            else:
                traj_length += 1
                state = state_


        # Update Policy

        # print('states ', states)
        # print('actions', actions)
        # print('dones  ', dones)
        # print('rewards', rewards)
        # print('values ', values)

        # Sample next value
        _,_,value = model(state)
        values.append(value)

        γ = 0.99
        returns = []
        R = value.detach().numpy()
        T = len(states) # trajectory length
        for t in reversed(range(T)):
            R = rewards[t] + dones[t]*γ*values[t+1].detach().numpy() + (1-dones[t])*γ*R
            returns.insert(0,R)

        values  = values[:-1]
        #returns = returns[:-1]

        #log_probs = torch.tensor(log_probs, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = returns - values

        policy_losses = []
        value_losses = []
        for log_prob, adv, value, r in zip(log_probs, advantages, values, returns):
            # print(log_prob, adv, -log_prob*adv)
            policy_losses.append(-log_prob*adv.detach())
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([r])))

        policy_loss = torch.stack(policy_losses).sum()
        value_loss = torch.stack(value_losses).sum()
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"epoch {epoch:3} -> steps {np.mean(traj_lengths):4.0f}  | rewards {torch.sum(rewards)/(len(traj_lengths)+1):12.4f} |   loss {loss:18.2f}  |   policy {policy_loss:12.2f}  |   value {value_loss:8.2f}")

        # print()
        # print()
        # print('###########################################')
        # print('states ', states)
        # print('actions', actions)
        # print('dones  ', dones)
        # print('rewards', rewards)
        # print('returns', returns)
        # print('values ', values)
        # print('advantages', advantages)
        #
        # print('states ', len(states))
        # print('actions', len(actions))
        # print('dones  ', len(dones))
        # print('rewards', len(rewards))
        # print('values ', len(values))
        # print('returns', len(returns))


if __name__ == '__main__':
    train()
