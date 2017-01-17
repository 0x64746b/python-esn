#!/usr/bin/env python
# coding: utf-8

"""Learn the Mackey-Glass equation."""


from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import matplotlib.pyplot as plt
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
    num_sampling_points = num_periods * samples_per_period

    frequencies = np.zeros(num_sampling_points)
    signal = np.zeros(num_sampling_points)

    sampling_points = np.linspace(
        0,
        num_periods * 2 * np.pi,
        num_sampling_points,
        endpoint=False
    )

    frequency_intervals = np.sort(np.append(
        [0, num_sampling_points],
        np.random.randint(0, num_sampling_points, num_frequency_changes)
    ))

    for i in range(num_frequency_changes + 1):
        frequency = np.random.randint(1, max_frequency + 1)

        frequencies[frequency_intervals[i]:frequency_intervals[i + 1]] = frequency
        signal[frequency_intervals[i]:frequency_intervals[i + 1]] = np.sin(
            frequency * sampling_points[frequency_intervals[i]:frequency_intervals[i + 1]]
        )

    return frequencies, signal


if __name__ == '__main__':
    input_frequencies, signal = generate_signal(
        NUM_PERIODS,
        SAMPLES_PER_PERIOD,
        NUM_FREQUENCY_CHANGES,
        MAX_FREQUENCY,
    )

    # plot some periods
    plt.plot(signal[:SAMPLES_PER_PERIOD * 30], label='Signal')
    plt.plot(
        [
            frequency/10
            for frequency in input_frequencies[:SAMPLES_PER_PERIOD * 30]
        ],
        label='Input frequency'
    )
    plt.title('First 30 periods')
    plt.show()