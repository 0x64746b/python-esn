# coding: utf-8

"""
Generate values from a starting point.

Manually feed back predicted values into an `Esn`.
"""

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import argparse
import logging

import numpy as np

from esn import Esn
from esn.examples import setup_logging
from esn.examples.mackey_glass import load_data, plot_results


logger = logging.getLogger(__name__)


def _generate(training_inputs, training_outputs, test_inputs, test_outputs):
    """Manually feedback predicted values into the inputs."""
    esn = Esn(
        in_size=1,
        reservoir_size=1000,
        out_size=1,
        spectral_radius=0.75,
        leaking_rate=0.3,
        sparsity=0.95,
        initial_transients=100,
        state_noise=1e-10,
    )

    # train
    esn.fit(training_inputs, training_outputs)

    # test
    predicted_outputs = [esn.predict(test_inputs[0])]
    for i in range(len(test_inputs)-1):
        predicted_outputs.append(esn.predict(predicted_outputs[i]))

    #  debug
    for i, predicted_date in enumerate([test_inputs[0]] + predicted_outputs[:-1]):
        logger.debug(
            '% f -> % f (Δ % f)',
            predicted_date,
            predicted_outputs[i],
            test_outputs[i] - predicted_outputs[i]
        )

    plot_results(
        test_outputs,
        predicted_outputs,
        mode='generate with manual feedback'
    )


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
    _generate(*data)
