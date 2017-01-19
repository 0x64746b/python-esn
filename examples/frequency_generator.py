#!/usr/bin/env python
# coding: utf-8

"""Learn the Mackey-Glass equation."""


from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import logging

from matplotlib import pyplot as plt, ticker
import numpy as np

from esn import ESN


SIGNAL_LENGTH = 15000
SAMPLES_PER_PERIOD = 200  # without endpoint
NUM_FREQUENCY_CHANGES = int(SIGNAL_LENGTH / 200)
MAX_FREQUENCY = 5

NUM_TRAINING_SAMPLES = int(SIGNAL_LENGTH * 0.7)


logger = logging.getLogger('python-esn.examples')


def generate_signal(
        num_sampling_points,
        samples_per_period,
        num_frequency_changes,
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
        np.random.randint(0, num_sampling_points, num_frequency_changes)
    ))

    for (start, end) in zip(frequency_intervals, frequency_intervals[1:]):
        frequencies[start:end] = np.random.randint(1, max_frequency + 1)

    sampling_point = 0
    for i in range(num_sampling_points):
        sampling_point += norm_sampling_distance * frequencies[i]
        signal[i] = np.sin(sampling_point)

    return frequencies, signal


def generate_data():
    frequencies, signal = generate_signal(
        SIGNAL_LENGTH,
        SAMPLES_PER_PERIOD,
        NUM_FREQUENCY_CHANGES,
        MAX_FREQUENCY,
    )

    training_inputs = np.array(zip(
        frequencies[:NUM_TRAINING_SAMPLES],
        signal[:NUM_TRAINING_SAMPLES])
    ).reshape(NUM_TRAINING_SAMPLES, 2, 1)
    # TODO: Check dimensionality. Do we *really* need this? Cf `correct_outputs`
    training_outputs = signal[None, 1:NUM_TRAINING_SAMPLES + 1]

    # consume training data
    frequencies = np.delete(frequencies, np.s_[:NUM_TRAINING_SAMPLES])
    signal = np.delete(signal, np.s_[:NUM_TRAINING_SAMPLES])

    inputs = np.array(zip(frequencies[:-1], signal[:-1])).reshape(len(frequencies[:-1]), 2, 1)
    correct_outputs = signal[1:]

    return training_inputs, training_outputs, inputs, correct_outputs


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    training_inputs, training_outputs, inputs, correct_outputs = generate_data()

    esn = ESN(
        in_size=2,
        reservoir_size=200,
        out_size=1,
        spectral_radius=0.25,
        leaking_rate=0.3,
        washout=100,
        smoothing_factor=0.0001
    )

    esn.fit(training_inputs, training_outputs)

    predicted_outputs = [esn.predict(input_date)[0][0] for input_date in inputs]

    # debug
    for i, input_date in enumerate(inputs):
        logger.debug(
            '% f -> % f (Δ % f)',
            input_date,
            predicted_outputs[i],
            correct_outputs[i] - predicted_outputs[i]
        )


    # plot some periods
    plt.plot(correct_outputs[:4000], label='Reference')
    plt.plot(predicted_outputs[:4000], label='Predicted')
    plt.plot(
        [
            input_date[0] / 10
            for input_date in inputs[:4000]
        ],
        label='Input frequency'
    )
    plt.gca().xaxis.set_major_locator(
        ticker.MultipleLocator(SAMPLES_PER_PERIOD)
    )
    plt.yticks([-1, -0.5, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 1])
    plt.title('Start of signal')
    plt.legend()
    plt.show()