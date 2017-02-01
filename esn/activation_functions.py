# coding: utf-8

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)


import numpy as np


def identity(x):
    return x


def lecun(x):
    return 1.7159 * np.tanh(2/3 * x)


def lecun_inv(x):
    return 3/2 * np.arctanh(x/1.7159)
