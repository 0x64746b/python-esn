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

import hyperopt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from esn import LmsEsn
from esn.activation_functions import lecun
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
        self.training_inputs = training_inputs
        self.training_outputs = training_outputs

        self.test_inputs = test_inputs
        self.test_outputs = test_outputs

    def run(self, output_file):
        np.random.seed(780245044)

        predicted_outputs = self._train(
            reservoir_size=3000,
            spectral_radius=1.31,
            leaking_rate=0.45,
            learning_rate=0.000011,
            sparsity=0.67,
            initial_transients=1000,
            state_noise=0.0023251,
            squared_network_state=True,
            activation_function=lecun,
            bias_scale=0.19,
            signal_scale=-3.4,
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

        plot_results(
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
            learning_rate,
            sparsity,
            initial_transients,
            state_noise,
            squared_network_state,
            activation_function,
            bias_scale,
            signal_scale,
            num_tracked_units=0,
    ):
        self.esn = LmsEsn(
            in_size=1,
            reservoir_size=int(reservoir_size),
            out_size=1,
            spectral_radius=spectral_radius,
            leaking_rate=leaking_rate,
            learning_rate=learning_rate,
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

    def optimize(self, exp_key):
        search_space = (
            hyperopt.hp.quniform('reservoir_size', 3000, 3001, 1000),
            hyperopt.hp.quniform('spectral_radius', 0.01, 2, 0.01),
            hyperopt.hp.quniform('leaking_rate', 0.01, 1, 0.01),
            hyperopt.hp.qloguniform('learning_rate', np.log(0.0000001), np.log(0.1), 0.0000001),
            hyperopt.hp.quniform('sparsity', 0.01, 0.99, 0.01),
            hyperopt.hp.quniform('initial_transients', 100, 10001, 100),
            hyperopt.hp.quniform('state_noise', 1e-7, 1e-2, 1e-7),
            hyperopt.hp.choice('squared_network_state', [False, True]),
            hyperopt.hp.choice('activation_function', [np.tanh, lecun]),
            hyperopt.hp.qnormal('bias_scale', 1, 1, 0.01),
            hyperopt.hp.qnormal('signal_scale', 1, 1, 0.1),
        )

        trials = hyperopt.mongoexp.MongoTrials(
            'mongo://localhost:27017/python_esn_trials/jobs',
            exp_key=exp_key,
        )

        best = hyperopt.fmin(
            self._objective,
            space=search_space,
            algo=hyperopt.tpe.suggest,
            max_evals=1000,
            trials=trials,
        )

        logger.info('Best parameter combination: %s', best)

    def _objective(self, hyper_parameters):
        # re-seed for repeatable results
        random_seed = np.random.randint(2**32)
        np.random.seed(random_seed)

        logger.debug(
            'seed: %s | sampled hyper-parameters: %s',
            random_seed,
            hyper_parameters
        )

        predicted_outputs = self._train(*hyper_parameters)
        try:
            rmse = np.sqrt(mean_squared_error(
                self.test_outputs,
                predicted_outputs
            ))
        except ValueError as error:
            return {'status': hyperopt.STATUS_FAIL, 'problem': str(error)}
        else:
            return {
                'status': hyperopt.STATUS_OK,
                'loss': rmse,
                'seed': str(random_seed)
            }
