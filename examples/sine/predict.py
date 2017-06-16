# coding: utf-8

"""Predict the next value for each given input with a `WienerHopfEsn`."""

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import logging

import numpy as np
import pandas as pd

from esn import WienerHopfEsn
from esn.activation_functions import lecun
from esn.examples import plot_results
from esn.examples.sine import SAMPLES_PER_PERIOD


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
        self.training_inputs = np.array(list(zip(*training_inputs)))
        self.training_outputs = np.array(training_outputs).reshape(
            len(training_outputs),
            1,  # out_size
        )
        self.test_inputs = np.array(list(zip(*test_inputs)))
        self.test_outputs = test_outputs

    def run(self, output_file):
        predicted_outputs = self._train()

        # debug
        for i, input_date in enumerate(self.test_inputs):
            logger.debug(
                '% f | % f -> % f (Δ % f)',
                input_date[0],
                input_date[1],
                predicted_outputs[i],
                self.test_outputs[i] - predicted_outputs[i]
            )

        plot_results(
            data=pd.DataFrame({
                'frequencies': self.test_inputs[:, 0],
                'correct outputs': self.test_outputs,
                'predicted outputs': predicted_outputs.flatten(),
            }),
            mode='predict',
            periodicity=SAMPLES_PER_PERIOD,
            output_file=output_file,
        )

    def _train(self):
        self.esn = WienerHopfEsn(
            in_size=2,
            reservoir_size=200,
            out_size=1,
            spectral_radius=0.25,
            leaking_rate=0.3,
            sparsity=0.95,
            initial_transients=100,
            ridge_regression=0.001,
            activation_function=lecun,
        )

        # train
        self.esn.fit(self.training_inputs, self.training_outputs)

        # test
        return np.array([
            self.esn.predict(input_date)
            for input_date in self.test_inputs
        ])
