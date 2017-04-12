# coding: utf-8

"""Generate values from a starting point using an `ESN` with output feedback."""

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import logging
import pickle

import hyperopt
import numpy as np
import scipy
from sklearn.metrics import mean_squared_error

from esn import Esn
from esn.activation_functions import lecun, lecun_inv
from esn.examples.sine import plot_results


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
        #  use only the frequency as input,
        #  the signal is fed back from the output
        self.training_inputs = np.array(training_inputs[0]).reshape(
            len(training_inputs[0]),
            1  # in_size
        )
        self.training_outputs = np.array(training_outputs).reshape(
            len(training_outputs),
            1  # out_size
        )
        self.test_inputs = np.array(test_inputs[0]).reshape(
            len(test_inputs[0]),
            1  # in_size
        )
        self.test_outputs = test_outputs

    def run(self):
        predicted_outputs = self._train(
            spectral_radius=0.25,
            leaking_rate=0.1,
            state_noise=0.007,
        )

        # debug
        for i, predicted_date in enumerate([0] + predicted_outputs[:-1]):
            logger.debug(
                '% f | % f -> % f (Δ % f)',
                self.test_inputs[i],
                predicted_date,
                predicted_outputs[i],
                self.test_outputs[i] - predicted_outputs[i]
            )

        plot_results(
            self.test_inputs,
            self.test_outputs,
            predicted_outputs,
            mode='generate with structural feedback'
        )

    def optimize(self, exp_key):
        def objective(hyper_parameters):
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

        search_space = (
            hyperopt.hp.quniform('spectral_radius', 0, 1.5, 0.01),
            hyperopt.hp.quniform('leaking_rate', 0, 1, 0.01),
            hyperopt.hp.quniform('state_noise', 0.0001, 0.1, 0.0001),
            hyperopt.hp.qnormal('bias_scale', 1, 1, 0.1),
            hyperopt.hp.qnormal('frequency_scale', 1, 1, 0.1),
        )

        trials = hyperopt.Trials(exp_key=exp_key)

        best = hyperopt.fmin(
            objective,
            space=search_space,
            algo=hyperopt.tpe.suggest,
            max_evals=150,
            trials=trials,
        )

        with open('{}_trials.pickle'.format(exp_key), 'wb') as trials_file:
            pickle.dump(trials, trials_file)

        with open('{}_best.pickle'.format(exp_key), 'wb') as result_file:
            pickle.dump(best, result_file)

        logger.info('Best parameter combination: %s', best)

    def _train(
            self,
            spectral_radius,
            leaking_rate,
            state_noise,
            bias_scale=1.0,
            frequency_scale=1.0,
    ):
        self.esn = Esn(
            in_size=1,
            reservoir_size=200,
            out_size=1,
            spectral_radius=spectral_radius,
            leaking_rate=leaking_rate,
            sparsity=0.95,
            initial_transients=1000,
            state_noise=state_noise,
            squared_network_state=True,
            activation_function=lecun,
            output_activation_function=(lecun, lecun_inv),
            output_feedback=True,
        )

        # scale input weights
        self.esn.W_in *= [bias_scale, frequency_scale]

        # train
        self.esn.fit(self.training_inputs, self.training_outputs)

        # test
        predicted_outputs = [self.esn.predict(self.test_inputs[0])[0]]
        for i in range(1, len(self.test_inputs)):
            predicted_outputs.append(self.esn.predict(self.test_inputs[i]))

        return predicted_outputs
