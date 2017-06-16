# coding: utf-8

"""Generate values from a starting point using an `Esn` with output feedback."""

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
import scipy
from sklearn.metrics import mean_squared_error

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

    def run(self, output_file):
        predicted_outputs = self._train(
            spectral_radius=1.15,
            leaking_rate=0.84,
            state_noise=5e-6,
        )

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
            output_file=output_file,
        )

    def _train(
            self,
            spectral_radius,
            leaking_rate,
            state_noise,
    ):
        self.esn = Esn(
            in_size=0,
            reservoir_size=1000,
            out_size=1,
            spectral_radius=spectral_radius,
            leaking_rate=leaking_rate,
            sparsity=0.95,
            initial_transients=100,
            state_noise=state_noise,
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

    def optimize(self, exp_key):
        search_space = (
            hyperopt.hp.quniform('spectral_radius', 0, 1.5, 0.01),
            hyperopt.hp.quniform('leaking_rate', 0, 1, 0.01),
            hyperopt.hp.quniform('state_noise', 1e-10, 1e-2, 1e-10),
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

    def _objective(self, hyper_parameters):
        # re-seed for repeatable results
        np.random.seed(48)

        try:
            predicted_outputs = self._train(*hyper_parameters)
        except scipy.sparse.linalg.ArpackNoConvergence:
            return {'status': hyperopt.STATUS_FAIL}
        else:
            try:
                rmse = np.sqrt(mean_squared_error(
                    self.test_outputs,
                    predicted_outputs
                ))
            except ValueError:
                return {'status': hyperopt.STATUS_FAIL}
            else:
                return {'status': hyperopt.STATUS_OK, 'loss': rmse}
