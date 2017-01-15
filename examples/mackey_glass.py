#!/usr/bin/env python
# coding: utf-8

"""Learn the Mackey-Glass equation."""


from __future__ import absolute_import, print_function, unicode_literals

import argparse
import logging

import numpy as np
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from esn import ESN


NUM_TRAINING_SAMPLES = 2000
NUM_PREDICTION_SAMPLES = 1000


logger = logging.getLogger('python-esn.examples')


def predict(training_inputs, training_outputs, inputs, correct_outputs):
    """Predict the next value for each given input."""

    def debug():
        for i, input_date in enumerate(inputs):
            logger.debug(
                '% f -> % f (Δ % f)',
                input_date,
                predicted_outputs[i],
                correct_outputs[i] - predicted_outputs[i]
            )

    esn = ESN(
        in_size=1,
        reservoir_size=1000,
        out_size=1,
        spectral_radius=1.25,
        leaking_rate=0.3,
        washout=100,
        smoothing_factor=0.0001
    )

    esn.fit(training_inputs, training_outputs)

    predicted_outputs = [esn.predict(input_date)[0][0] for input_date in inputs]

    debug()
    plot_results(correct_outputs, predicted_outputs, mode='predict')


def generate(training_inputs, training_outputs, inputs, correct_outputs):
    """Generate values from a starting point."""

    def debug():
        logger.debug(
            '% f -> % f (Δ % f)',
            inputs[0],
            predicted_outputs[0],
            correct_outputs[0] - predicted_outputs[0]
        )
        for i, predicted_date in enumerate(predicted_outputs[:-1]):
            logger.debug(
                '% f -> % f (Δ % f)',
                predicted_date,
                predicted_outputs[i+1],
                correct_outputs[i+1] - predicted_outputs[i+1]
            )

    esn = ESN(
        in_size=1,
        reservoir_size=1000,
        out_size=1,
        spectral_radius=1.25,
        leaking_rate=0.3,
        washout=100,
        smoothing_factor=0.0001
    )

    esn.fit(training_inputs, training_outputs)

    predicted_outputs = [esn.predict(inputs[0])[0][0]]
    for i in range(1, len(inputs)):
        predicted_outputs.append(esn.predict(predicted_outputs[i-1])[0][0])

    debug()
    plot_results(correct_outputs, predicted_outputs, mode='generate')


def plot_results(reference, predicted, mode):
    try:
        mse = mean_squared_error(reference, predicted)
    except ValueError as error:
        mse = error.message

    plt.plot(reference, label='Reference')
    plt.plot(predicted, label='Predicted')
    plt.gca().add_artist(AnchoredText('MSE: {}'.format(mse), loc=2))
    plt.gca().set_title('Mode: {}'.format(mode))
    plt.gca().set_ylim([-0.5, 0.5])
    plt.legend()
    plt.show()


def load_data(file_name):
    data = np.loadtxt(file_name)

    training_inputs = data[:NUM_TRAINING_SAMPLES]
    training_outputs = data[None, 1:NUM_TRAINING_SAMPLES+1]

    # consume training data
    data = np.delete(data, np.s_[:NUM_TRAINING_SAMPLES])

    inputs = data[:NUM_PREDICTION_SAMPLES]
    correct_outputs = data[1:NUM_PREDICTION_SAMPLES+1]

    return training_inputs, training_outputs, inputs, correct_outputs


def setup_logging(level):
    logging.basicConfig(
        format='%(name)s.%(module)s::%(funcName)s [%(levelname)s]: %(message)s',
        level=level.upper()
    )


def parse_command_line_args():
    main_command = argparse.ArgumentParser(description=__doc__)
    main_command.add_argument(
        '-l',
        '--log-level',
        metavar='LVL',
        default='info',
        choices=['debug', 'info', 'warning', 'error', 'critical'],
        help='The lowest level to log (default: %(default)s)'
    )
    sub_commands = main_command.add_subparsers(
        dest='sub_command',
        metavar='sub-command'
    )

    predict_command = sub_commands.add_parser(
        'predict',
        help=predict.__doc__
    )
    predict_command.add_argument(
        'data_file',
        help='the file containing the data to learn'
    )

    generate_command = sub_commands.add_parser(
        'generate',
        help=generate.__doc__
    )
    generate_command.add_argument(
        'data_file',
        help='the file containing the data to learn'
    )

    return main_command.parse_args()


COMMANDS = {
    'predict': predict,
    'generate': generate,
}


if __name__ == '__main__':
    args = parse_command_line_args()

    setup_logging(args.log_level)

    data = load_data(args.data_file)
    COMMANDS[args.sub_command](*data)
