# coding: utf-8

"""
Manually feed back predicted values into a `WienerHopfESN`
instead of using structural feedback.
"""

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import logging

import numpy as np

from esn import WienerHopfEsn
from esn.activation_functions import lecun
from esn.preprocessing import add_noise
from esn.examples.sine import plot_results


INPUT_NOISE_FACTOR = 0.03


logger = logging.getLogger(__name__)


class Example(object):

    @staticmethod
    def run(training_inputs, training_outputs, test_inputs, test_outputs):
        esn = WienerHopfEsn(
            in_size=2,
            reservoir_size=90,
            out_size=1,
            spectral_radius=0.25,
            leaking_rate=0.1,
            sparsity=0.95,
            initial_transients=1000,
            ridge_regression=0.001,
            squared_network_state=True,
            activation_function=lecun,
        )

        # format data
        #  add noise to the signal to help stabilize the amplitude
        training_inputs = np.array(list(zip(
            training_inputs[0],
            add_noise(training_inputs[1], INPUT_NOISE_FACTOR)
        )))
        training_outputs = np.array(training_outputs).reshape(
            len(training_outputs),
            esn.L
        )
        test_inputs = np.array(list(zip(*test_inputs)))

        # train
        esn.fit(training_inputs, training_outputs)

        # test
        predicted_outputs = [esn.predict(test_inputs[0])[0]]
        for i in range(1, len(test_inputs)):
            next_input = np.array([test_inputs[i][0], predicted_outputs[i - 1]])
            predicted_outputs.append(esn.predict(next_input)[0])

        #  debug
        for i, predicted_date in enumerate([test_inputs[0][1]] + predicted_outputs[:-1]):
            logger.debug(
                '% f | % f -> % f (Δ % f)',
                test_inputs[i][0],
                predicted_date,
                predicted_outputs[i],
                test_outputs[i] - predicted_outputs[i]
            )

        plot_results(
            test_inputs[:, 0],
            test_outputs,
            predicted_outputs,
            mode='generate with manual feedback'
        )
