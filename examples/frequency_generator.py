#!/usr/bin/env python
# coding: utf-8

"""Learn a sine signal with a varying frequency."""


from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import argparse
import logging

from matplotlib import pyplot as plt, ticker
from matplotlib.offsetbox import AnchoredText
import numpy as np
from sklearn.metrics import mean_squared_error

from esn import ESN
from esn.activation_functions import lecun
from esn.preprocessing import scale


SIGNAL_LENGTH = 15000
SAMPLES_PER_PERIOD = 300  # without endpoint
NUM_FREQUENCY_CHANGES = int(SIGNAL_LENGTH / 200)
MAX_FREQUENCY = 5

NOISE_FACTOR = 0.03
NUM_TRAINING_SAMPLES = int(SIGNAL_LENGTH * 0.7)


logger = logging.getLogger('python-esn.examples')


def predict(training_inputs, training_outputs, inputs, correct_outputs):
    """Predict the next value for each given input."""

    esn = ESN(
        in_size=2,
        reservoir_size=200,
        out_size=1,
        spectral_radius=0.25,
        leaking_rate=0.3,
        washout=100,
        ridge_regression=0.001,
        activation_function=lecun,
    )

    # format data
    training_inputs = np.array(zip(*training_inputs))
    training_outputs = np.array(training_outputs).reshape(
        len(training_outputs),
        esn.L,
    )
    inputs = np.array(zip(*inputs))

    # train
    esn.fit(training_inputs, training_outputs)

    # test
    predicted_outputs = [esn.predict(input_date)[0] for input_date in inputs]

    #  debug
    for i, input_date in enumerate(inputs):
        logger.debug(
            '% f | % f -> % f (Δ % f)',
            input_date[0],
            input_date[1],
            predicted_outputs[i],
            correct_outputs[i] - predicted_outputs[i]
        )

    plot_results(inputs[:, 0], correct_outputs, predicted_outputs, mode='predict')


def generate(training_inputs, training_outputs, inputs, correct_outputs):
    """Generate values from a starting point."""

    esn = ESN(
        in_size=2,
        reservoir_size=200,
        out_size=1,
        spectral_radius=0.25,
        leaking_rate=0.1,
        washout=1000,
        ridge_regression=0.001,
        activation_function=lecun,
    )

    # format data
    #  add noise to the signal to help stabilize the amplitude
    training_inputs = np.array(zip(
        training_inputs[0],
        training_inputs[1]
        + np.random.normal(0, 1, len(training_inputs[1])) * NOISE_FACTOR
    ))
    training_outputs = np.array(training_outputs).reshape(
        len(training_outputs),
        esn.L
    )
    inputs = np.array(zip(*inputs))

    # train
    esn.fit(training_inputs, training_outputs)

    # test
    predicted_outputs = [esn.predict(inputs[0])[0]]
    for i in range(1, len(inputs)):
        next_input = np.array([inputs[i][0], predicted_outputs[i-1]])
        predicted_outputs.append(esn.predict(next_input)[0])

    #  debug
    for i, predicted_date in enumerate([inputs[0][1]] + predicted_outputs[:-1]):
        logger.debug(
            '% f | % f -> % f (Δ % f)',
            inputs[i][0],
            predicted_date,
            predicted_outputs[i],
            correct_outputs[i] - predicted_outputs[i]
        )

    plot_results(inputs[:, 0], correct_outputs, predicted_outputs, mode='generate')


def plot_results(frequencies, correct_outputs, predicted_outputs, mode):
    try:
        mse = mean_squared_error(correct_outputs, predicted_outputs)
    except ValueError as error:
        mse = error.message
    plt.plot(
        frequencies,
        color='r',
        label='Input frequency'
    )
    plt.plot(correct_outputs, label='Correct outputs')
    plt.plot(predicted_outputs, label='Predicted outputs')
    plt.gca().xaxis.set_major_locator(
        ticker.MultipleLocator(SAMPLES_PER_PERIOD)
    )
    plt.gca().add_artist(AnchoredText('MSE: {}'.format(mse), loc=2))
    plt.title('Mode: {}'.format(mode))
    plt.legend()
    plt.show()


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


def load_data():
    frequencies, signal = generate_signal(
        SIGNAL_LENGTH,
        SAMPLES_PER_PERIOD,
        NUM_FREQUENCY_CHANGES,
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

    inputs = (frequencies[:-1], signal[:-1])
    correct_outputs = signal[1:]

    return training_inputs, training_outputs, inputs, correct_outputs


def setup_logging(verbosity):
    logging.basicConfig(
        format='%(name)s.%(module)s::%(funcName)s [%(levelname)s]: %(message)s',
        level=max(logging.DEBUG, logging.WARNING - verbosity * 10)
    )


def parse_command_line_args():
    main_command = argparse.ArgumentParser(description=__doc__)
    main_command.add_argument(
        '-v',
        '--verbose',
        dest='verbosity',
        action='count',
        default=0,
        help='Increase the log level with each use'
    )
    sub_commands = main_command.add_subparsers(
        dest='sub_command',
        metavar='sub-command'
    )

    sub_commands.add_parser(
        'predict',
        help=predict.__doc__
    )

    sub_commands.add_parser(
        'generate',
        help=generate.__doc__
    )

    return main_command.parse_args()


COMMANDS = {
    'predict': predict,
    'generate': generate,
}


if __name__ == '__main__':
    args = parse_command_line_args()

    setup_logging(args.verbosity)

    data = load_data()
    COMMANDS[args.sub_command](*data)
