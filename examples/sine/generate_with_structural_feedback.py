# coding: utf-8

"""Generate values from a starting point using an `ESN` with output feedback."""

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
from esn.activation_functions import lecun, lecun_inv
from esn.examples import setup_logging
from esn.examples.sine import load_data, plot_results


logger = logging.getLogger(__name__)


def _generate(training_inputs, training_outputs, test_inputs, test_outputs):
    esn = Esn(
        in_size=1,
        reservoir_size=200,
        out_size=1,
        spectral_radius=0.25,
        leaking_rate=0.1,
        sparsity=0.95,
        initial_transients=1000,
        state_noise=0.007,
        squared_network_state=True,
        activation_function=lecun,
        output_activation_function=(lecun, lecun_inv),
        output_feedback=True,
    )

    # format data
    #  use only the frequency as input, the signal is fed back from the output
    training_inputs = np.array(training_inputs[0]).reshape(
        len(training_inputs[0]),
        esn.K
    )
    training_outputs = np.array(training_outputs).reshape(
        len(training_outputs),
        esn.L
    )
    test_inputs = np.array(test_inputs[0]).reshape(len(test_inputs[0]), esn.K)

    # train
    esn.fit(training_inputs, training_outputs)

    # test
    predicted_outputs = [esn.predict(test_inputs[0])[0]]
    for i in range(1, len(test_inputs)):
        predicted_outputs.append(esn.predict(test_inputs[i]))

    #  debug
    for i, predicted_date in enumerate([0] + predicted_outputs[:-1]):
        logger.debug(
            '% f | % f -> % f (Δ % f)',
            test_inputs[i],
            predicted_date,
            predicted_outputs[i],
            test_outputs[i] - predicted_outputs[i]
        )

    plot_results(
        test_inputs,
        test_outputs,
        predicted_outputs,
        mode='generate with structural feedback'
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
    args = parser.parse_args()

    setup_logging(args.verbosity)

    # explicitly seed PRNG for comparable runs
    np.random.seed(48)

    data = load_data()
    _generate(*data)
