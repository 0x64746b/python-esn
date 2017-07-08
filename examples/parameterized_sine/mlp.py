# coding: utf-8

"""Train an ESN prepended to a multilayer perceptron."""

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import logging

import hyperopt
import hyperopt.mongoexp
import numpy as np
import pandas as pd

from esn import MlpEsn
from esn.activation_functions import lecun
from esn.preprocessing import add_noise
from esn.examples import EsnExample
from esn.examples.parameterized_sine import SAMPLES_PER_PERIOD


INPUT_NOISE_FACTOR = 0.03


logger = logging.getLogger(__name__)


class MlpExample(EsnExample):

    search_space = (
        hyperopt.hp.quniform('spectral_radius', 0, 1.5, 0.01),
        hyperopt.hp.quniform('leaking_rate', 0, 1, 0.01),
        hyperopt.hp.qnormal('bias_scale', 1, 1, 0.01),
        hyperopt.hp.qnormal('frequency_scale', 1, 1, 0.01),
        hyperopt.hp.qnormal('signal_scale', 1, 1, 0.1),
    )

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

    def run(self, output_file):
        predicted_outputs = self._train(
            spectral_radius=1.5,
            leaking_rate=0.1,
            bias_scale=2.6,
            frequency_scale=2.2,
            signal_scale=5.5,
        )

        # debug
        for i, predicted_date in enumerate([self.test_inputs[0][1]] + predicted_outputs[:-1]):
            logger.debug(
                '% f | % f -> % f (Δ % f)',
                self.test_inputs[i][0],
                predicted_date,
                predicted_outputs[i],
                self.test_outputs[i] - predicted_outputs[i]
            )

        self._plot_results(
            data=pd.DataFrame({
                'Frequencies': self.test_inputs[:, 0],
                'Correct outputs': self.test_outputs,
                'Predicted outputs': predicted_outputs,
            }),
            title='Generate with manual feedback',
            periodicity=SAMPLES_PER_PERIOD,
            output_file=output_file,
        )

    def _train(
            self,
            spectral_radius,
            leaking_rate,
            bias_scale=1.0,
            frequency_scale=1.0,
            signal_scale=1.0,
            num_tracked_units=0,
    ):
        self.esn = MlpEsn(
            in_size=2,
            reservoir_size=200,
            out_size=1,
            spectral_radius=spectral_radius,
            leaking_rate=leaking_rate,
            sparsity=0.95,
            initial_transients=1000,
            squared_network_state=True,
            activation_function=lecun,
        )
        self.esn.num_tracked_units = num_tracked_units

        # scale input weights
        self.esn.W_in *= [bias_scale, frequency_scale, signal_scale]

        # train
        self.esn.fit(self.training_inputs, self.training_outputs)

        # test
        S = [np.hstack((
            self.esn.BIAS,
            self.test_inputs[0],
            self.esn.x,
            self.test_inputs[0]**2,
            self.esn.x**2))
        ]
        predicted_outputs = [self.esn.predict(self.test_inputs[0])[0]]
        for i in range(1, len(self.test_inputs)):
            next_input = np.array([self.test_inputs[i][0], predicted_outputs[i - 1]])
            predicted_outputs.append(self.esn.predict(next_input)[0])
            S.append(np.hstack((
                self.esn.BIAS,
                self.test_inputs[i],
                self.esn.x,
                self.test_inputs[i]**2,
                self.esn.x**2))
            )

        if num_tracked_units:
            self.test_activations = self.esn.track_most_influential_units(
                np.array(S)
            )

        return predicted_outputs
