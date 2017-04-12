# coding: utf-8

"""
Manually feed back predicted values into an `Esn`
instead of using structural feedback.
"""

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import logging

from esn import Esn
from esn.examples.mackey_glass import plot_results


logger = logging.getLogger(__name__)


class Example(object):

    def __init__(
            self,
            training_inputs,
            training_outputs,
            test_inputs,
            test_outputs
    ):
        self.training_inputs = training_inputs
        self.training_outputs = training_outputs
        self.test_inputs = test_inputs
        self.test_outputs = test_outputs

    def run(self):
        predicted_outputs = self._train()

        # debug
        for i, predicted_date in enumerate([self.test_inputs[0]] + predicted_outputs[:-1]):
            logger.debug(
                '% f -> % f (Δ % f)',
                predicted_date,
                predicted_outputs[i],
                self.test_outputs[i] - predicted_outputs[i]
            )

        plot_results(
            self.test_outputs,
            predicted_outputs,
            mode='generate with manual feedback',
            tracked_activations=self.esn.tracked_units,
        )

    def _train(self):
        self.esn = Esn(
            in_size=1,
            reservoir_size=1000,
            out_size=1,
            spectral_radius=0.75,
            leaking_rate=0.3,
            sparsity=0.95,
            initial_transients=100,
            state_noise=1e-10,
        )
        self.esn.num_tracked_units = 2

        # train
        self.esn.fit(self.training_inputs, self.training_outputs)

        # test
        predicted_outputs = [self.esn.predict(self.test_inputs[0])]
        for i in range(len(self.test_inputs) - 1):
            predicted_outputs.append(self.esn.predict(predicted_outputs[i]))

        return predicted_outputs
