# coding: utf-8

"""Learn a sine wave with changing frequency."""


from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import numpy as np

from esn.examples import EsnExample
from esn.preprocessing import add_noise, scale


NUM_PERIODS = 10000
SAMPLES_PER_PERIOD = 300  # without endpoint
NUM_SAMPLING_POINTS = NUM_PERIODS * SAMPLES_PER_PERIOD
MAX_FREQUENCY = 5
INPUT_NOISE_FACTOR = 0.03


class FrequencyGeneratorExample(EsnExample):

    def __init__(self):
        super(FrequencyGeneratorExample, self).__init__()
        self.periodicity = SAMPLES_PER_PERIOD

    @staticmethod
    def generate_signal():
        """
        Generate a sine signal with varying frequency.

        Inspired by https://github.com/cknd/pyESN/blob/master/freqgen.ipynb.
        """
        norm_sampling_distance = 2 * np.pi / SAMPLES_PER_PERIOD

        frequencies = np.zeros(NUM_SAMPLING_POINTS)
        signal = np.zeros(NUM_SAMPLING_POINTS)

        frequency_intervals = np.sort(np.append(
            [0, NUM_SAMPLING_POINTS],
            np.random.randint(
                0,
                NUM_SAMPLING_POINTS,
                int(NUM_SAMPLING_POINTS / SAMPLES_PER_PERIOD)
            )
        ))

        for (start, end) in zip(frequency_intervals, frequency_intervals[1:]):
            frequencies[start:end] = np.random.randint(1, MAX_FREQUENCY + 1)

        sampling_point = 0
        for i in range(NUM_SAMPLING_POINTS):
            sampling_point += norm_sampling_distance * frequencies[i]
            signal[i] = np.sin(sampling_point)

        return frequencies, signal

    def _load_data(self, offset=0):
        # explicitly seed PRNG for reproducible data generation
        np.random.seed(42)

        frequencies, signal = self.generate_signal()

        # shift to a fresh set of data
        discarded = offset * (self.num_training_samples + self.num_test_samples)
        frequencies = np.delete(frequencies, np.s_[:discarded])
        signal = np.delete(signal, np.s_[:discarded])

        # scale frequencies to [-1, 1]
        frequencies = scale(frequencies)

        self.training_inputs = np.array(list(zip(
            frequencies[:self.num_training_samples],
            add_noise(signal[:self.num_training_samples], INPUT_NOISE_FACTOR)
        )))
        self.training_outputs = signal[1:self.num_training_samples + 1].reshape(
            self.num_training_samples,
            1
        )

        # consume training data
        frequencies = np.delete(frequencies, np.s_[:self.num_training_samples])
        signal = np.delete(signal, np.s_[:self.num_training_samples])

        self.test_inputs = np.array(list(zip(
            frequencies[:self.num_test_samples],
            signal[:self.num_test_samples]
        )))
        self.test_outputs = signal[1:self.num_test_samples + 1]


# make modules importable from the package name space.
#  import late to break cyclic import
from .pseudoinverse import PseudoinverseExample
from .lms import LmsExample
from .mlp import MlpExample
