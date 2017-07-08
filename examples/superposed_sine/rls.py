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
import hyperopt.mongoexp
import numpy as np
import pandas as pd

from esn import RlsEsn
from esn.activation_functions import lecun
from esn.examples import EsnExample
from esn.preprocessing import add_noise, scale


logger = logging.getLogger(__name__)


class RlsExample(EsnExample):

    def __init__(self, *data):
        super(RlsExample, self).__init__(*data)

        # remove every other label
        self.training_outputs[1::2] = np.nan


    def run(self, output_file):
        predicted_outputs = self._train(
            spectral_radius=1.11,
            leaking_rate=0.75,
            forgetting_factor=0.99998,
            autocorrelation_init=0.1,
            bias_scale=-0.4,
            signal_scale=1.2,
            state_noise=0.004,
            input_noise=0.007,
        )

        # debug
        for i, predicted_date in enumerate([self.test_inputs[0]] + predicted_outputs[:-1]):
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
            title='Generate simple signal',
            output_file=output_file,
        )

    def _train(
            self,
            spectral_radius,
            leaking_rate,
            forgetting_factor,
            autocorrelation_init,
            bias_scale,
            signal_scale,
            state_noise,
            input_noise,
    ):
        self.esn = RlsEsn(
            in_size=1,
            reservoir_size=1000,
            out_size=1,
            spectral_radius=spectral_radius,
            leaking_rate=leaking_rate,
            forgetting_factor=forgetting_factor,
            autocorrelation_init=autocorrelation_init,
            state_noise=state_noise,
            sparsity=0.95,
            initial_transients=300,
            squared_network_state=True,
            activation_function=lecun,
        )
        self.esn.W_in *= [bias_scale, signal_scale]

        # train
        self.esn.fit(
            np.array([self.training_inputs[0]]),
            np.array([self.training_outputs[0]])
        )
        for input_date, output_date in zip(
                add_noise(self.training_inputs[1:], input_noise),
                self.training_outputs[1:]
        ):
            if not np.isnan(output_date.item()):
                self.esn.partial_fit(
                    np.array([input_date]),
                    np.array([output_date])
                )
            else:
                # drive reservoir
                self.esn.predict(input_date)

        # test
        predicted_outputs = [self.esn.predict(self.test_inputs[0])]
        for i in range(len(self.test_inputs)-1):
            predicted_outputs.append(self.esn.predict(predicted_outputs[i]))

        return np.array(predicted_outputs)

    def optimize(self, exp_key):
        search_space = (
            hyperopt.hp.quniform('spectral_radius', 0, 1.5, 0.01),
            hyperopt.hp.quniform('leaking_rate', 0, 1, 0.01),
            hyperopt.hp.quniform('forgetting_factor', 0.98, 1, 0.0001),
            hyperopt.hp.qloguniform('autocorrelation_init', np.log(0.1), np.log(1), 0.0001),
            hyperopt.hp.qnormal('bias_scale', 1, 1, 0.1),
            hyperopt.hp.qnormal('signal_scale', 1, 1, 0.1),
            hyperopt.hp.quniform('state_noise', 1e-10, 1e-2, 1e-10),
            hyperopt.hp.quniform('input_noise', 1e-10, 1e-2, 1e-10),
        )

        trials = hyperopt.mongoexp.MongoTrials(
            'mongo://localhost:27017/python_esn_trials/jobs',
            exp_key=exp_key,
        )

        best = hyperopt.fmin(
            self._objective,
            space=search_space,
            algo=hyperopt.tpe.suggest,
            max_evals=150,
            trials=trials,
        )

        logger.info('Best parameter combination: %s', best)
