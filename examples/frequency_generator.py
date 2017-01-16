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
SAMPLES_PER_PERIOD = 201
MAX_FREQUENCY = 5
MIN_PERIOD = 2
MAX_PERIOD = 10


def generate_signal(
        num_periods,
        samples_per_period,
        max_frequency,
        min_period,
        max_period
):
    frequencies = []
    signal = []

    target_length = num_periods * samples_per_period

    while len(signal) < target_length:
        frequency = np.random.randint(1, max_frequency + 1)
        section_length = min(
            np.random.randint(min_period, max_period + 1),
            target_length - len(signal)
        )

        print('Frequency {} over {} periods'.format(frequency, section_length))

        sampling_points = np.linspace(
            0,
            2 * np.pi * section_length,
            samples_per_period * section_length
        )
        section = np.sin(frequency * sampling_points[:-1])
        signal.extend(section)
        frequencies.extend([frequency] * samples_per_period * section_length)

    return frequencies, signal


if __name__ == '__main__':
    input_frequencies, signal = generate_signal(
        NUM_PERIODS,
        SAMPLES_PER_PERIOD,
        MAX_FREQUENCY,
        MIN_PERIOD,
        MAX_PERIOD,
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