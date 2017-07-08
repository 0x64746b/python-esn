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
from esn.examples import EsnExample


logger = logging.getLogger(__name__)


class Example(EsnExample):

    def run(self, output_file):
        predicted_outputs = self._train()

        # debug
        for i, input_date in enumerate(self.test_inputs):
            logger.debug(
                '% f -> % f (Δ % f)',
                input_date,
                predicted_outputs[i],
                self.test_outputs[i] - predicted_outputs[i]
            )

        self._plot_results(
            data=pd.DataFrame({
                'Correct outputs': self.test_outputs,
                'Predicted outputs': predicted_outputs.flatten(),
            }),
            title='Predict',
            output_file=output_file,
        )

    def _train(self):

        self.esn = WienerHopfEsn(
            in_size=1,
            reservoir_size=1000,
            out_size=1,
            spectral_radius=1.25,
            leaking_rate=0.3,
            sparsity=0.95,
            initial_transients=100,
            ridge_regression=0.0001
        )

        self.esn.fit(self.training_inputs, self.training_outputs)

        return np.array([
            self.esn.predict(input_date)
            for input_date in self.test_inputs
        ])

