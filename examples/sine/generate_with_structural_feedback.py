# coding: utf-8

"""Generate values from a starting point using an `ESN` with output feedback."""

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import logging

import numpy as np

from esn import Esn
from esn.activation_functions import lecun, lecun_inv
from esn.examples.sine import plot_results


logger = logging.getLogger(__name__)


class Example(object):

    def __init__(
            self,
            training_inputs,
            training_outputs,
            test_inputs,
            test_outputs
    ):
        # format data
        #  use only the frequency as input,
        #  the signal is fed back from the output
        self.training_inputs = np.array(training_inputs[0]).reshape(
            len(training_inputs[0]),
            1  # in_size
        )
        self.training_outputs = np.array(training_outputs).reshape(
            len(training_outputs),
            1  # out_size
        )
        self.test_inputs = np.array(test_inputs[0]).reshape(
            len(test_inputs[0]),
            1  # in_size
        )
        self.test_outputs = test_outputs

    def run(self):
        predicted_outputs = self._train()

        # debug
        for i, predicted_date in enumerate([0] + predicted_outputs[:-1]):
            logger.debug(
                '% f | % f -> % f (Δ % f)',
                self.test_inputs[i],
                predicted_date,
                predicted_outputs[i],
                self.test_outputs[i] - predicted_outputs[i]
            )

        plot_results(
            self.test_inputs,
            self.test_outputs,
            predicted_outputs,
            mode='generate with structural feedback'
        )

    def _train(self):
        self.esn = Esn(
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

        # train
        self.esn.fit(self.training_inputs, self.training_outputs)

        # test
        predicted_outputs = [self.esn.predict(self.test_inputs[0])[0]]
        for i in range(1, len(self.test_inputs)):
            predicted_outputs.append(self.esn.predict(self.test_inputs[i]))

        return predicted_outputs
