import torch
import numpy as np


def gaussian_policy(φ, θ, Σ, deterministic=False):
    """
    A parameterized guassian policy.
    :param φ: features
    :param θ: parameters for the mean
    :param Σ: covariance matrix
    :param deterministic: if True, then the policy behaves deterministically, i.e. it just returns the mean
    :return: sampled action from a parameterized, multivariate gaussian policy.
    """
    if not deterministic:
        return np.random.multivariate_normal(mean=φ.T @ θ, cov=Σ)
    else:
        return φ.T @ θ


def save_tb_scalars(writer, epoch, **kwargs):
    summary_string = f"{epoch:4}"
    for metric in kwargs:
        summary_string += f"  |  {metric} {kwargs[metric]:16.6}"
        writer.add_scalar(f'rl/{metric}', torch.tensor(kwargs[metric], dtype=torch.float32), epoch)
    print(summary_string)
