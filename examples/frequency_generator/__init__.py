# coding: utf-8

"""Learn a sine wave with changing frequency."""


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

MAX_FREQUENCY = 5

NUM_TRAINING_SAMPLES = SAMPLES_PER_PERIOD * 30
NUM_TEST_SAMPLES = SAMPLES_PER_PERIOD * 15


def generate_signal(
        num_sampling_points,
        samples_per_period,
        max_frequency,
):
    """
    Generate a sine signal with varying frequency.

    Inspired by https://github.com/cknd/pyESN/blob/master/freqgen.ipynb.
    """
    norm_sampling_distance = 2 * np.pi / samples_per_period

    frequencies = np.zeros(num_sampling_points)
    signal = np.zeros(num_sampling_points)

    frequency_intervals = np.sort(np.append(
        [0, num_sampling_points],
        np.random.randint(
            0,
            num_sampling_points,
            int(num_sampling_points/samples_per_period)
        )
    ))

    for (start, end) in zip(frequency_intervals, frequency_intervals[1:]):
        frequencies[start:end] = np.random.randint(1, max_frequency + 1)

    sampling_point = 0
    for i in range(num_sampling_points):
        sampling_point += norm_sampling_distance * frequencies[i]
        signal[i] = np.sin(sampling_point)

    return frequencies, signal


def load_data():
    frequencies, signal = generate_signal(
        NUM_SAMPLING_POINTS,
        SAMPLES_PER_PERIOD,
        MAX_FREQUENCY,
    )

    # scale frequencies to [-1, 1]
    frequencies = scale(frequencies)

    training_inputs = (
        frequencies[:NUM_TRAINING_SAMPLES],
        signal[:NUM_TRAINING_SAMPLES]
    )
    training_outputs = signal[1:NUM_TRAINING_SAMPLES + 1]

    # consume training data
    frequencies = np.delete(frequencies, np.s_[:NUM_TRAINING_SAMPLES])
    signal = np.delete(signal, np.s_[:NUM_TRAINING_SAMPLES])

    inputs = (
        frequencies[:NUM_TEST_SAMPLES],
        signal[:NUM_TEST_SAMPLES]
    )
    correct_outputs = signal[1:NUM_TEST_SAMPLES + 1]

    return training_inputs, training_outputs, inputs, correct_outputs


# make modules importable from the package name space.
#  import late to break cyclic import
from .pseudoinverse import PseudoinverseExample
from .mlp import MlpExample
