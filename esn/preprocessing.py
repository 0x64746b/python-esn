# coding: utf-8

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)


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
