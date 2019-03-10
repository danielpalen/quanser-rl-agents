# Application of Reinforcement Learning Methods

This repository implements a set of reinforcement learning algorithms to run on OpenAI gym environments.
The algorihtms implemented are:
- [Relative Entropy Policy Search (REPS)](https://www.ias.informatik.tu-darmstadt.de/uploads/Team/JanPeters/Peters2010_REPS.pdf)
- [Actor-Critic Relative Entropy Policy Search (ACREPS)](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12247)
- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347)

In practice they were only tested on the following environments:
- Pendulum Swingup
- Double Cartpole
- Furuta Pendulum
- Ball Ballancer

The last three are custom gym environments implemented in the quansar_robots repository.

## Repository Structure
/quanser-rl-agents
* /agents
    * acreps.py
    * reps.py
    * ppo.py
* /common
* /hyperparameters
* /out
* experiments.py - convenience method to run multiple experiments with the same settings in parallel processes.
* run.py - main entry point for all training and evaluation tasks.

## Training and Running Models
The main entry point of this repository is the `run.py` file. It comes with a sophisticated command-line parser and
special subparsers for each algorithm implemented in this repository.
The basic syntax is:
```
python run.py [general arguments] (ACREPS|REPS|PPO) [algorithm specific arguments]
```
More information on required and on optional commands can be explored by running `python run.py -h`.
Which returns
```
python run.py [-h] --name NAME --env ENV [--robot] [--n_epochs N_EPOCHS]
              [--n_steps N_STEPS] [--seed SEED] [--render] [--experiment]
              [--eval | --resume]
              {REPS,ACREPS,PPO}
```
For information on algorithm specific commands the ``-h`` can executed on the subcommands `{ACREPS,REPS,PPO}`,
e.g. ``python run.py REPS -h`` to get more information training the REPS algorithm.

The most basic command for running REPS on the underactuated pendulum swingup would be
```
python run.py --name reps_pendulum --env pendulum REPS
```

#### Visualising Training with TensorBoard
TODO: Explain where tensorboard files get saved and how to open them.

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

```
cd clients
pip install -e .
```

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

### Other requirements
- joblib
- yaml
- ...

### Installing the learning algorithms

```git clone https://github.com/danielpalen/rl-research-lab-class.git```

Now you are all set to train a model or evaluate an algorithm.

## Run Experiments
...
