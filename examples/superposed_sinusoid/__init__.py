# coding: utf-8

"""Learn a superposed sine signal."""


from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import numpy as np

from esn.examples import EsnExample
from esn.preprocessing import scale


NUM_PERIODS = 300
SAMPLES_PER_PERIOD = 300  # without endpoint
NUM_SAMPLING_POINTS = NUM_PERIODS * SAMPLES_PER_PERIOD


class SuperposedSinusoidExample(EsnExample):

    def _load_data(self):
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

        self.training_inputs = signal[:self.num_training_samples]
        self.training_outputs = signal[1:self.num_training_samples + 1].copy()

        # consume training data
        signal = np.delete(signal, np.s_[:self.num_training_samples], axis=0)

        self.test_inputs = signal[:self.num_test_samples]
        self.test_outputs = signal[1:self.num_test_samples + 1]


# make modules importable from the package name space.
#  import late to break cyclic import
from .pseudoinverse import PseudoinverseExample
from .rls import RlsExample
