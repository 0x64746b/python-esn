# coding: utf-8

"""Predict the next value for each given input with a `WienerHopfEsn`."""

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)


import argparse
import logging

import numpy as np

from esn import WienerHopfEsn
from esn.examples import setup_logging
from esn.examples.mackey_glass import load_data, plot_results


logger = logging.getLogger(__name__)


def _predict(training_inputs, training_outputs, test_inputs, test_outputs):
    esn = WienerHopfEsn(
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

    predicted_outputs = [esn.predict(input_date) for input_date in test_inputs]

    # debug
    for i, input_date in enumerate(test_inputs):
        logger.debug(
            '% f -> % f (Δ % f)',
            input_date,
            predicted_outputs[i],
            test_outputs[i] - predicted_outputs[i]
        )

    plot_results(test_outputs, predicted_outputs, mode='predict')


def main():
    """The main entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-v',
        '--verbose',
        dest='verbosity',
        action='count',
        default=0,
        help='Increase the log level with each use'
    )
    parser.add_argument(
        'data_file',
        help='the file containing the data to learn'
    )
    args = parser.parse_args()

    setup_logging(args.verbosity)

    # explicitly seed PRNG for comparable runs
    np.random.seed(48)

    data = load_data(args.data_file)
    _predict(*data)
