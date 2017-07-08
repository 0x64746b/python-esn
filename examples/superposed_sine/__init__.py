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


NUM_PERIODS = 1000
SAMPLING_RATE = 50  # points per period
NUM_SAMPLING_POINTS = NUM_PERIODS * SAMPLING_RATE

TRAINING_LENGTH = int(NUM_SAMPLING_POINTS * 0.7)
TEST_LENGTH = 500


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

    training_inputs = signal[:TRAINING_LENGTH]
    training_outputs = signal[1:TRAINING_LENGTH + 1].copy()

    # consume training data
    signal = np.delete(signal, np.s_[:TRAINING_LENGTH])

    test_inputs = signal[:TEST_LENGTH]
    test_outputs = signal[1:TEST_LENGTH + 1]

    return training_inputs, training_outputs, test_inputs, test_outputs


# make modules importable from the package name space.
#  import late to break cyclic import
from .rls import RlsExample
