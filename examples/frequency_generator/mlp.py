# coding: utf-8

"""
Use the extended system states of an ESN as inputs to a multilayer perceptron.
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

from esn import MlpEsn
from esn.activation_functions import lecun
from . import FrequencyGeneratorExample


logger = logging.getLogger(__name__)


class MlpExample(FrequencyGeneratorExample):

    def __init__(self):
        super(MlpExample, self).__init__()

        self.num_training_samples = 850000
        self.num_test_samples = 5000

        self.title = 'Frequency generator; MLP; {} samples'.format(
            self.num_training_samples
        )

        self.random_seed = 3637692427
        self.hyper_parameters = {
            'reservoir_size': 200,
            'spectral_radius': 0.92,
            'leaking_rate': 0.09,
            'sparsity': 0.4,
            'initial_transients': 100,
            'state_noise': 0.0001132,
            'squared_network_state': True,
            'activation_function': lecun,
            'mlp_hidden_layer_size': 400,
            'mlp_activation_function': 'relu',
            'mlp_solver': 'adam',
            'bias_scale': -0.3,
            'frequency_scale': 1.48,
            'signal_scale': -2.97,
        }

        self.search_space_choices.update({
            'mlp_activation_function': ['identity', 'logistic', 'tanh', 'relu'],
            'mlp_solver': ['sgd', 'adam'],
        })

        self.search_space = (
            hyperopt.hp.quniform('reservoir_size', 200, 201, 10),
            hyperopt.hp.quniform('spectral_radius', 0, 1.5, 0.01),
            hyperopt.hp.quniform('leaking_rate', 0.01, 1, 0.01),
            hyperopt.hp.quniform('sparsity', 0, 0.99, 0.1),
            hyperopt.hp.quniform('initial_transients', 100, 501, 50),
            hyperopt.hp.quniform('state_noise', 1e-7, 1e-3, 1e-7),
            hyperopt.hp.choice(*self._build_choice('squared_network_state')),
            hyperopt.hp.choice(*self._build_choice('activation_function')),
            hyperopt.hp.quniform('mlp_hidden_layer_size', 50, 501, 50),
            hyperopt.hp.choice(*self._build_choice('mlp_activation_function')),
            hyperopt.hp.choice(*self._build_choice('mlp_solver')),
            hyperopt.hp.qnormal('bias_scale', 1, 1, 0.01),
            hyperopt.hp.qnormal('frequency_scale', 1, 1, 0.01),
            hyperopt.hp.qnormal('signal_scale', 1, 1, 0.01),
        )

    def _train(
            self,
            reservoir_size,
            spectral_radius,
            leaking_rate,
            sparsity,
            initial_transients,
            state_noise,
            squared_network_state,
            activation_function,
            mlp_hidden_layer_size,
            mlp_activation_function,
            mlp_solver,
            bias_scale,
            frequency_scale,
            signal_scale,
            num_tracked_units=0,
    ):
        self.esn = MlpEsn(
            in_size=2,
            reservoir_size=int(reservoir_size),
            out_size=1,
            spectral_radius=spectral_radius,
            leaking_rate=leaking_rate,
            sparsity=sparsity,
            initial_transients=int(initial_transients),
            state_noise=state_noise,
            squared_network_state=squared_network_state,
            activation_function=activation_function,
            mlp_hidden_layer_sizes=(int(mlp_hidden_layer_size),),
            mlp_activation_function=mlp_activation_function,
            mlp_solver=mlp_solver,
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
        data = super(MlpExample, self)._get_plotting_data(predicted_outputs)
        data['Frequencies'] = self.test_inputs[:, 0]
        return data
