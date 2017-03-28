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
from esn.preprocessing import scale


NUM_TRAINING_SAMPLES = 2000
NUM_PREDICTION_SAMPLES = 1000


logger = logging.getLogger('python-esn.examples')


def predict(training_inputs, training_outputs, inputs, correct_outputs):
    """Predict the next value for each given input."""

    esn = ESN(
        in_size=1,
        reservoir_size=1000,
        out_size=1,
        spectral_radius=1.25,
        leaking_rate=0.3,
        sparsity=0.95,
        initial_transients=100,
        ridge_regression=0.0001
    )

    esn.fit(training_inputs, training_outputs)

    predicted_outputs = [esn.predict(input_date)[0] for input_date in inputs]

    # debug
    for i, input_date in enumerate(inputs):
        logger.debug(
            '% f -> % f (Δ % f)',
            input_date,
            predicted_outputs[i],
            correct_outputs[i] - predicted_outputs[i]
        )

    plot_results(correct_outputs, predicted_outputs, mode='predict')


def generate(training_inputs, training_outputs, inputs, correct_outputs):
    """Generate values from a starting point."""

    esn = ESN(
        in_size=0,
        reservoir_size=1000,
        out_size=1,
        spectral_radius=0.75,
        leaking_rate=0.3,
        sparsity=0.95,
        initial_transients=100,
        ridge_regression=0.0001,
        output_feedback=True,
    )

    # create "no" inputs
    training_inputs = np.array([[]] * len(training_inputs))
    inputs = np.array([[]] * len(inputs))

    esn.fit(training_inputs, training_outputs)

    predicted_outputs = [esn.predict(input_date)[0] for input_date in inputs]

    # debug
    for i, predicted_date in enumerate([training_outputs[-1]] + predicted_outputs[:-1]):
        logger.debug(
            '% f -> % f (Δ % f)',
            predicted_date,
            predicted_outputs[i],
            correct_outputs[i] - predicted_outputs[i]
        )

    plot_results(correct_outputs, predicted_outputs, mode='generate')


def plot_results(reference, predicted, mode):
    try:
        rmse = np.sqrt(mean_squared_error(reference, predicted))
    except ValueError as error:
        rmse = error.message

    plt.plot(reference, label='Reference')
    plt.plot(predicted, label='Predicted')
    plt.gca().add_artist(AnchoredText('RMSE: {}'.format(rmse), loc=2))
    plt.gca().set_title('Mode: {}'.format(mode))
    plt.legend()
    plt.show()


def load_data(file_name):
    data = np.loadtxt(file_name)

    # scale data to stretch to [-1, 1]
    data = scale(data)

    training_inputs = data[:NUM_TRAINING_SAMPLES].reshape(
        NUM_TRAINING_SAMPLES,
        1  # in_size
    )
    training_outputs = data[1:NUM_TRAINING_SAMPLES+1].reshape(
        NUM_TRAINING_SAMPLES,
        1  # out_size
    )

    # consume training data
    data = np.delete(data, np.s_[:NUM_TRAINING_SAMPLES])

    inputs = data[:NUM_PREDICTION_SAMPLES].reshape(NUM_PREDICTION_SAMPLES, 1)
    correct_outputs = data[1:NUM_PREDICTION_SAMPLES+1]

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


def main():
    """The main entry point."""
    args = parse_command_line_args()

    setup_logging(args.verbosity)

    # explicitly seed PRNG for comparable runs
    np.random.seed(48)

    data = load_data(args.data_file)
    COMMANDS[args.sub_command](*data)


if __name__ == '__main__':
    main()
