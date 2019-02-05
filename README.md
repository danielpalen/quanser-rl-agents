# Application of Reinforcement Learning Methods

This repository implements a set of reinforcement learning algorithms to run on OpenAI gym environments.
The algorihtms implemented are:
- Relative Entropy Policy Search (REPS)
- Actor-Critic Relative Entropy Policy Search (ACREPS)
- Proximal Policy Optimization (PPO)

In practice they were only tested on the following environments:
- Pendulum Swingup
- Double Cartpole
- Furuta Pendulum
- Ball Ballancer
The last three are custom gym environments implemented in the quansar robots repository.

## Installation

In order to run the code provided, follow these steps to install the necessary components.

### Conda environment

In order to manage the different python versions we use conda for creating virtual environments. First, get mini conda from <https://conda.io/en/latest/miniconda.html> and install it. Next, create an empty virtual conda environment with Python 3. Create the environment by executing:

```conda create --name rl-env python=3.6.5```

and then activate it by executing

```conda activate rl-env```

**IMPORTANT: When not using a virtual environment, use *pip3* instead of *pip* in the instructions below.**

### Quanser Robots
Install the modified OpenAI gym environments by first cloning the git repository

```git clone https://git.ias.informatik.tu-darmstadt.de/quanser/clients.git```

and then install by executing

```cd clients ; pip install -e .```

### Pytorch framework
As we used the pytorch framework (<https://pytorch.org/#pip-install-pytorch>) install the appropriate version:

For mac:

```pip install torch torchvision```

For Linux (non-gpu version):

```
pip install https://download.pytorch.org/whl/cpu/torch-1.0.0-cp36-cp36m-linux_x86_64.whl
pip install torchvision
```

### Tensorflow and TensorboardX

In order to install tensorflow (non-gpu version), execute:

```pip install tensorflow```

For evaluation and visualization of the learning, install tensorboardX, which is a tensorboard port for Pytorch:

```pip install tensorboardX```

### Installing the learning algorithms

```git clone https://github.com/danielpalen/rl-research-lab-class.git```

Now you are all set to train a model or evaluate an algorithm.

## Run Experiments
...
