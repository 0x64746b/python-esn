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


NUM_TRAINING_SAMPLES = 1000
NUM_PREDICTION_SAMPLES = 1000


logger = logging.getLogger('python-esn.examples')


class SubCommand(object):

    def __init__(self, training_inputs, training_outputs, inputs):
        self._training_inputs = training_inputs
        self._training_outputs = training_outputs
        self._inputs = inputs

    def plot(self, reference, predicted):
        self._debug(reference, predicted)

        mse = mean_squared_error(reference, predicted)

        plt.plot(reference, label='Reference')
        plt.plot(predicted, label='Predicted')
        plt.gca().add_artist(AnchoredText('MSE: {}'.format(mse), loc=2))
        plt.gca().set_title('Mode: {}'.format(self.__class__.__name__))
        plt.legend()
        plt.show()


class Predictor(SubCommand):
    """Predict the next value for each given input."""

    def __call__(self):
        esn = ESN(
            in_size=1,
            reservoir_size=1000,
            out_size=1,
            spectral_radius=1.25,
            leaking_rate=0.3,
            washout=100,
            smoothing_factor=0.0001
        )
        esn.fit(self._training_inputs, self._training_outputs)

        return [esn.predict(input_date)[0][0] for input_date in self._inputs]

    def _debug(self, reference, predicted):
        for i, input_date in enumerate(self._inputs):
            logger.debug(
                '% f -> % f (Δ % f)',
                input_date,
                predicted[i],
                reference[i] - predicted[i]
            )


def load_data(file_name):
    data = np.loadtxt(file_name)

    training_inputs = data[:NUM_TRAINING_SAMPLES]
    training_outputs = data[None, 1:NUM_TRAINING_SAMPLES+1]

    np.delete(data, np.s_[:NUM_TRAINING_SAMPLES])

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

    predict_command = sub_commands.add_parser('predict', help=Predictor.__doc__)
    predict_command.add_argument(
        'data_file',
        help='the file containing the data to learn'
    )

    return main_command.parse_args()


COMMANDS = {
    'predict': Predictor,
}


if __name__ == '__main__':
    args = parse_command_line_args()

    setup_logging(args.log_level)

    data = load_data(args.data_file)
    command = COMMANDS[args.sub_command](*data[:-1])

    results = command()

    command.plot(data[-1], results)
