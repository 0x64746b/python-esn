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
from . import MackeyGlassExample


logger = logging.getLogger(__name__)


class MlpExample(MackeyGlassExample):

    def __init__(self, *args, **kwargs):
        super(MlpExample, self).__init__(*args, **kwargs)

        self.num_training_samples = 150000
        self.num_test_samples = 500

        self.title = 'Mackey-Glass; MLP; {} samples'.format(
            self.num_training_samples
        )

        self.random_seed = 1919597375
        self.hyper_parameters = {
            'reservoir_size': 100,
            'spectral_radius': 1.43,
            'leaking_rate': 0.22,
            'sparsity': 0.33,
            'initial_transients': 450,
            'state_noise': 0.004151,
            'squared_network_state': False,
            'activation_function': np.tanh,
            'mlp_hidden_layer_size': 500,
            'mlp_activation_function': 'relu',
            'mlp_solver': 'adam',
            'bias_scale': -0.55,
            'signal_scale': -2.4,
        }

        self.search_space_choices.update({
            'mlp_activation_function': ['identity', 'logistic', 'tanh', 'relu'],
            'mlp_solver': ['sgd', 'adam'],
        })

        self.search_space = (
            hyperopt.hp.quniform('reservoir_size', 100, 101, 10),
            hyperopt.hp.quniform('spectral_radius', 0.01, 2, 0.01),
            hyperopt.hp.quniform('leaking_rate', 0.01, 1, 0.01),
            hyperopt.hp.quniform('sparsity', 0.01, 0.99, 0.01),
            hyperopt.hp.quniform('initial_transients', 100, 501, 50),
            hyperopt.hp.quniform('state_noise', 1e-7, 1e-2, 1e-7),
            hyperopt.hp.choice(*self._build_choice('squared_network_state')),
            hyperopt.hp.choice(*self._build_choice('activation_function')),
            hyperopt.hp.quniform('mlp_hidden_layer_size', 50, 501, 50),
            hyperopt.hp.choice(*self._build_choice('mlp_activation_function')),
            hyperopt.hp.choice(*self._build_choice('mlp_solver')),
            hyperopt.hp.qnormal('bias_scale', 1, 1, 0.01),
            hyperopt.hp.qnormal('signal_scale', 1, 1, 0.1),
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
            signal_scale,
            num_tracked_units=0,
    ):
        self.esn = MlpEsn(
            in_size=1,
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
        self.esn.W_in *= [bias_scale, signal_scale]

        # train
        self.esn.fit(self.training_inputs, self.training_outputs)

        # test
        predicted_outputs = [self.esn.predict(self.test_inputs[0])]
        for i in range(len(self.test_inputs) - 1):
            predicted_outputs.append(self.esn.predict(predicted_outputs[i]))

        return np.array(predicted_outputs)
