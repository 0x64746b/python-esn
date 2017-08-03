# coding: utf-8

"""Learn a superposed sine signal."""


from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import numpy as np

from esn.preprocessing import scale


NUM_PERIODS = 100
SAMPLES_PER_PERIOD = 300  # without endpoint
NUM_SAMPLING_POINTS = NUM_PERIODS * SAMPLES_PER_PERIOD

NUM_TRAINING_SAMPLES = SAMPLES_PER_PERIOD * 30
NUM_TEST_SAMPLES = SAMPLES_PER_PERIOD * 2


def load_data():
    sampling_points = np.linspace(
        0,
        NUM_PERIODS * 2 * np.pi,
        num=NUM_SAMPLING_POINTS
    )
    signal = scale(
        np.sin(sampling_points)
        + np.sin(2 * sampling_points)
        + np.sin(3.3 * sampling_points)
        + np.sin(4 * sampling_points)
        + np.cos(2.2 * sampling_points)
        + np.cos(4 * sampling_points)
        + np.cos(5 * sampling_points)
    ).reshape(NUM_SAMPLING_POINTS, 1)

    training_inputs = signal[:NUM_TRAINING_SAMPLES]
    training_outputs = signal[1:NUM_TRAINING_SAMPLES + 1].copy()

    # consume training data
    signal = np.delete(signal, np.s_[:NUM_TRAINING_SAMPLES], axis=0)

    test_inputs = signal[:NUM_TEST_SAMPLES]
    test_outputs = signal[1:NUM_TEST_SAMPLES + 1]

    return training_inputs, training_outputs, test_inputs, test_outputs


# make modules importable from the package name space.
#  import late to break cyclic import
from .pseudoinverse import PseudoinverseExample
from .rls import RlsExample
