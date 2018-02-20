# coding: utf-8

"""
Use the pseudoinverse of the extended system states to compute the output
weights.
"""

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

from esn import LmsEsn
from esn.activation_functions import lecun
from . import FrequencyGeneratorExample


logger = logging.getLogger(__name__)


class LmsExample(FrequencyGeneratorExample):

    def __init__(self):
        super(LmsExample, self).__init__()

        self.num_loops = 9
        self.num_training_samples = 100000
        self.num_test_samples = 5000

        self.title = 'Frequency generator; LMS; {} samples'.format(
            self.num_training_samples
        )

        self.random_seed = 626566608
        self.hyper_parameters = {
            'reservoir_size': 100,
            'spectral_radius': 1.05,
            'leaking_rate': 0.18,
            'learning_rate': 0.00004,
            'sparsity': 0.1,
            'initial_transients': 200,
            'state_noise': 0.00026,
            'squared_network_state': True,
            'activation_function': lecun,
            'bias_scale': -1.85,
            'frequency_scale': 1.5,
            'signal_scale': 1.97,
        }

        self.search_space = (
            hyperopt.hp.quniform('reservoir_size', 100, 101, 10),
            hyperopt.hp.quniform('spectral_radius', 0.01, 2, 0.01),
            hyperopt.hp.quniform('leaking_rate', 0.01, 1, 0.01),
            hyperopt.hp.qloguniform('learning_rate', np.log(0.00001), np.log(0.1), 0.00001),
            hyperopt.hp.quniform('sparsity', 0, 0.99, 0.1),
            hyperopt.hp.quniform('initial_transients', 100, 501, 50),
            hyperopt.hp.quniform('state_noise', 1e-7, 1e-3, 1e-7),
            hyperopt.hp.choice(*self._build_choice('squared_network_state')),
            hyperopt.hp.choice(*self._build_choice('activation_function')),
            hyperopt.hp.qnormal('bias_scale', 1, 1, 0.01),
            hyperopt.hp.qnormal('frequency_scale', 1, 1, 0.01),
            hyperopt.hp.qnormal('signal_scale', 1, 1, 0.01),
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
            frequency_scale,
            signal_scale,
            num_tracked_units=0,
    ):
        self.esn = LmsEsn(
            in_size=2,
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
        self.esn.W_in *= [bias_scale, frequency_scale, signal_scale]

        # train
        self.esn.fit(self.training_inputs, self.training_outputs)
        for i in range(1, self.num_loops):
            self.esn._num_seen_inputs = 0
            self.esn.partial_fit(self.training_inputs, self.training_outputs)

        # test
        test_states = [np.hstack((
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
            test_states.append(np.hstack((
                self.esn.BIAS,
                self.test_inputs[i],
                self.esn.x,
                self.test_inputs[i]**2,
                self.esn.x**2))
            )

        if num_tracked_units:
            self.test_activations = self.esn.track_most_influential_units(
                np.array(test_states)
            )

        return np.array(predicted_outputs)

    def _log_debug(self, predicted_outputs):
        for i, predicted_date in enumerate([self.test_inputs[0][1]] + predicted_outputs[:-1]):
            logger.debug(
                '% f | % f -> % f (Δ % f)',
                self.test_inputs[i][0],
                predicted_date,
                predicted_outputs[i],
                self.test_outputs[i] - predicted_outputs[i]
            )

    def _get_plotting_data(self, predicted_outputs):
        data = super(LmsExample, self)._get_plotting_data(predicted_outputs)
        data['Frequencies'] = self.test_inputs[:, 0]
        return data
