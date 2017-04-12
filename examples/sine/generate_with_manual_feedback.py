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

    def __init__(
            self,
            training_inputs,
            training_outputs,
            test_inputs,
            test_outputs
    ):
        self.training_inputs = np.array(list(zip(
            training_inputs[0],
            add_noise(training_inputs[1], INPUT_NOISE_FACTOR)
        )))
        self.training_outputs = np.array(training_outputs).reshape(
            len(training_outputs),
            1
        )
        self.test_inputs = np.array(list(zip(*test_inputs)))
        self.test_outputs = test_outputs

    def run(self):
        predicted_outputs = self._train()

        # debug
        for i, predicted_date in enumerate([self.test_inputs[0][1]] + predicted_outputs[:-1]):
            logger.debug(
                '% f | % f -> % f (Δ % f)',
                self.test_inputs[i][0],
                predicted_date,
                predicted_outputs[i],
                self.test_outputs[i] - predicted_outputs[i]
            )

        plot_results(
            self.test_inputs[:, 0],
            self.test_outputs,
            predicted_outputs,
            mode='generate with manual feedback'
        )

    def _train(self):
        self.esn = WienerHopfEsn(
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

        # train
        self.esn.fit(self.training_inputs, self.training_outputs)

        # test
        predicted_outputs = [self.esn.predict(self.test_inputs[0])[0]]
        for i in range(1, len(self.test_inputs)):
            next_input = np.array([self.test_inputs[i][0], predicted_outputs[i - 1]])
            predicted_outputs.append(self.esn.predict(next_input)[0])

        return predicted_outputs
