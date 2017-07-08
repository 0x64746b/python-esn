# coding: utf-8

"""Train an ESN with a recursive least squares filter."""

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import logging

import hyperopt
import numpy as np
import pandas as pd

from esn import RlsEsn
from esn.activation_functions import lecun
from esn.examples import EsnExample


logger = logging.getLogger(__name__)


class RlsExample(EsnExample):

    search_space = (
        hyperopt.hp.quniform('reservoir_size', 3000, 3001, 1000),
        hyperopt.hp.quniform('spectral_radius', 0.01, 2, 0.01),
        hyperopt.hp.quniform('leaking_rate', 0.01, 1, 0.01),
        hyperopt.hp.quniform('forgetting_factor', 0.98, 1, 0.0001),
        hyperopt.hp.qloguniform('autocorrelation_init', np.log(0.1), np.log(1), 0.0001),
        hyperopt.hp.quniform('sparsity', 0.01, 0.99, 0.01),
        hyperopt.hp.quniform('initial_transients', 100, 1001, 100),
        hyperopt.hp.quniform('state_noise', 1e-7, 1e-2, 1e-7),
        hyperopt.hp.choice('squared_network_state', [False, True]),
        hyperopt.hp.choice('activation_function', [np.tanh, lecun]),
        hyperopt.hp.qnormal('bias_scale', 1, 1, 0.01),
        hyperopt.hp.qnormal('signal_scale', 1, 1, 0.1),
    )

    def run(self, output_file):
        np.random.seed(1839385064)

        predicted_outputs = self._train(
            reservoir_size=3000,
            spectral_radius=1.14,
            leaking_rate=0.28,
            sparsity=0.66,
            initial_transients=5000,
            state_noise=0.0047857,
            squared_network_state=True,
            activation_function=lecun,
            bias_scale=0.88,
            signal_scale=3.3,
        )

        # debug
        for i, predicted_date in enumerate(np.concatenate((
                [self.test_inputs[0]],
                predicted_outputs[:-1])
        )):
            logger.debug(
                '% f -> % f (Δ % f)',
                predicted_date,
                predicted_outputs[i],
                self.test_outputs[i] - predicted_outputs[i]
            )

        self._plot_results(
            data=pd.DataFrame({
                'Correct outputs': self.test_outputs,
                'Predicted outputs': predicted_outputs.flatten(),
            }),
            title='Generate with manual feedback',
            output_file=output_file,
        )

    def _train(
            self,
            reservoir_size,
            spectral_radius,
            leaking_rate,
            forgetting_factor,
            autocorrelation_init,
            sparsity,
            initial_transients,
            state_noise,
            squared_network_state,
            activation_function,
            bias_scale,
            signal_scale,
            num_tracked_units=0,
    ):
        self.esn = RlsEsn(
            in_size=1,
            reservoir_size=int(reservoir_size),
            out_size=1,
            spectral_radius=spectral_radius,
            leaking_rate=leaking_rate,
            forgetting_factor=forgetting_factor,
            autocorrelation_init=autocorrelation_init,
            sparsity=sparsity,
            initial_transients=int(initial_transients),
            state_noise=state_noise,
            squared_network_state=squared_network_state,
            activation_function=activation_function,
        )
        self.esn.num_tracked_units = num_tracked_units

        # scale input weights
        self.esn.W_in *= [bias_scale, signal_scale]

        # train
        self.esn.fit(self.training_inputs, self.training_outputs)

        # test
        predicted_outputs = [self.esn.predict(self.test_inputs[0])]
        for i in range(len(self.test_inputs) - 1):
            predicted_outputs.append(self.esn.predict(predicted_outputs[i]))

        return np.array(predicted_outputs)
