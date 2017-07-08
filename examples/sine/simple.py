# coding: utf-8

"""Generate an unparametrized sine signal"""


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
import scipy
from sklearn.metrics import mean_squared_error

from esn import RlsEsn
from esn.activation_functions import lecun
from esn.examples import plot_results
from esn.preprocessing import add_noise, scale


logger = logging.getLogger(__name__)


class Example(object):

    def __init__(self):
        # generate data
        num_periods = 1000
        sampling_rate = 50  # points per period
        num_sampling_points = num_periods * sampling_rate

        training_length = int(num_sampling_points * 0.7)
        test_length = 500

        sampling_points = np.linspace(
            0,
            num_periods * 2 * np.pi,
            num=num_sampling_points
        )
        signal = scale(
            np.sin(sampling_points)
            + np.sin(2 * sampling_points)
            + np.sin(3.3 * sampling_points)
            + np.sin(4 * sampling_points)
            + np.cos(2.2 * sampling_points)
            + np.cos(4 * sampling_points)
            + np.cos(5 * sampling_points)
        ).reshape(num_sampling_points, 1)

        self.training_inputs = signal[:training_length]
        self.training_outputs = signal[1:training_length + 1].copy()

        # remove every other label
        self.training_outputs[1::2] = np.nan

        # consume training data
        signal = np.delete(signal, np.s_[:training_length])

        self.test_inputs = signal[:test_length]
        self.test_outputs = signal[1:test_length + 1]

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

        plot_results(
            data=pd.DataFrame({
                'correct outputs': self.test_outputs,
                'predicted outputs': predicted_outputs.flatten(),
            }),
            mode='generate simple signal',
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
