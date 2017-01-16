#!/usr/bin/env python
# coding: utf-8

"""Learn the Mackey-Glass equation."""


from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

from itertools import chain, repeat

import matplotlib.pyplot as plt
import numpy as np


MAX_FREQUENCY = 5
NUM_PERIODS = 50
NUM_SAMPLING_POINTS = 201


def generate_signal(num_periods, num_sampling_points, max_frequency):
    half_frequencies = [
        np.random.randint(1, max_frequency+1)
        for i in range(int(num_periods/2))
    ]

    # make sure each frequency persists for at least 2 periods
    frequencies = list(chain.from_iterable(
        zip(half_frequencies, half_frequencies)
    ))

    signal = []
    for frequency in frequencies:
        sampling_points = np.linspace(0, 2 * np.pi, num_sampling_points)
        period = np.sin(frequency * sampling_points[:-1])
        signal.extend(period)

    return frequencies, signal



if __name__ == '__main__':
    input_frequencies, signal = generate_signal(
        NUM_PERIODS,
        NUM_SAMPLING_POINTS,
        MAX_FREQUENCY
    )
    print('input frequency per period:', input_frequencies)

    # plot first 10 periods
    plt.plot(signal[:NUM_SAMPLING_POINTS * 10], label='Signal')
    plt.plot(list(chain.from_iterable(repeat(e, NUM_SAMPLING_POINTS) for e in [frequency / 10 for frequency in input_frequencies]))[:NUM_SAMPLING_POINTS * 10], label='Input frequency')
    plt.title('First 10 periods')
    plt.show()
