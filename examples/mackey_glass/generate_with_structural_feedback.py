# coding: utf-8

"""Generate values from a starting point using an `Esn` with output feedback."""

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import logging

import numpy as np

from esn import Esn
from esn.examples.mackey_glass import plot_results


logger = logging.getLogger(__name__)


def run(training_inputs, training_outputs, test_inputs, test_outputs):
    esn = Esn(
        in_size=0,
        reservoir_size=1000,
        out_size=1,
        spectral_radius=0.75,
        leaking_rate=0.3,
        sparsity=0.95,
        initial_transients=100,
        state_noise=1e-10,
        output_feedback=True,
    )

    # create "no" inputs
    training_inputs = np.array([[]] * len(training_inputs))
    test_inputs = np.array([[]] * len(test_inputs))

    esn.fit(training_inputs, training_outputs)

    # predict "no" inputs
    predicted_outputs = [esn.predict(input_date)[0] for input_date in test_inputs]

    # debug
    for i, predicted_date in enumerate([training_outputs[-1]] + predicted_outputs[:-1]):
        logger.debug(
            '% f -> % f (Δ % f)',
            predicted_date,
            predicted_outputs[i],
            test_outputs[i] - predicted_outputs[i]
        )

    plot_results(
        test_outputs,
        predicted_outputs,
        mode='generate with structural feedback'
    )
