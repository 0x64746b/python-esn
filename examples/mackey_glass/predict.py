# coding: utf-8

"""Predict the next value for each given input with a `WienerHopfEsn`."""

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)


import logging

from esn import WienerHopfEsn
from esn.examples.mackey_glass import plot_results


logger = logging.getLogger(__name__)


def run(training_inputs, training_outputs, test_inputs, test_outputs):
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
