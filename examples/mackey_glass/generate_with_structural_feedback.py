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
import pandas as pd

from esn import Esn
from esn.examples import plot_results


logger = logging.getLogger(__name__)


class Example(object):

    def __init__(
            self,
            training_inputs,
            training_outputs,
            test_inputs,
            test_outputs
    ):
        # create "no" inputs
        self.training_inputs = np.array([[]] * len(training_inputs))
        self.test_inputs = np.array([[]] * len(test_inputs))

        self.training_outputs = training_outputs
        self.test_outputs = test_outputs

    def run(self):
        predicted_outputs = self._train()

        # debug
        for i, predicted_date in enumerate([self.training_outputs[-1]] + predicted_outputs[:-1]):
            logger.debug(
                '% f -> % f (Δ % f)',
                predicted_date,
                predicted_outputs[i],
                self.test_outputs[i] - predicted_outputs[i]
            )

        plot_results(
            data=pd.DataFrame({
                'correct outputs': self.test_outputs,
                'predicted outputs': predicted_outputs.flatten(),
            }),
            mode='generate with structural feedback',
            debug={
                'training_activations': self.esn.tracked_units,
                'w_out': self.esn.W_out,
            },
        )

    def _train(self):
        self.esn = Esn(
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
        self.esn.num_tracked_units=5

        # train
        self.esn.fit(self.training_inputs, self.training_outputs)

        # predict "no" inputs
        predicted_outputs = [
            self.esn.predict(input_date)
            for input_date in self.test_inputs
        ]

        return np.array(predicted_outputs)
