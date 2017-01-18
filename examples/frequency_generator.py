#!/usr/bin/env python
# coding: utf-8

"""Learn the Mackey-Glass equation."""


from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

from matplotlib import pyplot as plt, ticker
import numpy as np


NUM_PERIODS = 50
SAMPLES_PER_PERIOD = 200  # without endpoint
NUM_FREQUENCY_CHANGES = 25
MAX_FREQUENCY = 5


def generate_signal(
        num_periods,
        samples_per_period,
        num_frequency_changes,
        max_frequency,
):
    """
    Generate a sine signal with varying frequency.

    Inspired by https://github.com/cknd/pyESN/blob/master/freqgen.ipynb.
    """
    num_sampling_points = num_periods * samples_per_period
    norm_sampling_distance = 2 * np.pi / samples_per_period

    frequencies = np.zeros(num_sampling_points)
    signal = np.zeros(num_sampling_points)

    frequency_intervals = np.sort(np.append(
        [0, num_sampling_points],
        np.random.randint(0, num_sampling_points, num_frequency_changes)
    ))

    for (start, end) in zip(frequency_intervals, frequency_intervals[1:]):
        frequencies[start:end] = np.random.randint(1, max_frequency + 1)

    sampling_point = 0
    for i in range(num_sampling_points):
        sampling_point += norm_sampling_distance * frequencies[i]
        signal[i] = np.sin(sampling_point)

    return frequencies, signal


if __name__ == '__main__':
    input_frequencies, signal = generate_signal(
        NUM_PERIODS,
        SAMPLES_PER_PERIOD,
        NUM_FREQUENCY_CHANGES,
        MAX_FREQUENCY,
    )

    # plot some periods
    plt.plot(signal[:4000], label='Signal')
    plt.plot(
        [
            frequency/10
            for frequency in input_frequencies[:4000]
        ],
        label='Input frequency'
    )
    plt.gca().xaxis.set_major_locator(
        ticker.MultipleLocator(SAMPLES_PER_PERIOD)
    )
    plt.yticks([-1, -0.5, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 1])
    plt.title('Start of signal')
    plt.show()