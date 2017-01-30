# coding: utf-8

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)


import numpy as np


def scale(samples, bound=1):
    """
    Scale the given samples to [-bound, bound].

    The scaled samples are always centered around 0.

    :param samples: The samples to be scaled
    :param bound: The bounds of the scaled samples
    :return: The scaled samples
    """
    # normalize to [0, 1]
    normalized = (samples - min(samples)) / (max(samples) - min(samples))

    # center around 0 by shifting to [-0.5, 0.5]
    centered = normalized - 0.5

    # scale to requested interval
    factor = bound / max(centered)
    return factor * centered


def add_noise(samples, size, mu=0, sigma=1):
    """
    Add noise of the given size to the given samples.

    The random noise is drawn from a normal distribution with the given center
    and standard deviation and then scaled by the given size.

    :param samples: The samples to add noise to
    :param size: The amount of noise to add
    :param mu: The mean of the noise
    :param sigma: The standard deviation of the noise
    :return: The noisy samples
    """
    return samples + np.random.normal(mu, sigma, samples.shape) * size
