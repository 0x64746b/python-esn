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

from esn import Esn
from esn.activation_functions import lecun
from . import SuperposedSinusoidExample


logger = logging.getLogger(__name__)


class PseudoinverseExample(SuperposedSinusoidExample):

    def __init__(self):
        super(PseudoinverseExample, self).__init__()

        self.num_training_samples = 10000
        self.num_test_samples = 500

        self.title = 'Superposed sine; Pseudoinverse; {} samples'.format(
            self.num_training_samples
        )

        self.random_seed = 1462542195
        self.hyper_parameters = {
            'reservoir_size': 200,
            'spectral_radius': 0.86,
            'leaking_rate': 0.08,
            'sparsity': 0.95,
            'initial_transients': 300,
            'state_noise': 0.0000003,
            'squared_network_state': True,
            'activation_function': lecun,
            'bias_scale': 0.3,
            'signal_scale': 0.6,
        }

        self.search_space = (
            hyperopt.hp.quniform('reservoir_size', 200, 201, 10),
            hyperopt.hp.quniform('spectral_radius', 0.01, 2, 0.01),
            hyperopt.hp.quniform('leaking_rate', 0.01, 1, 0.01),
            hyperopt.hp.quniform('sparsity', 0.01, 0.99, 0.01),
            hyperopt.hp.quniform('initial_transients', 100, 501, 50),
            hyperopt.hp.quniform('state_noise', 1e-7, 1e-2, 1e-7),
            hyperopt.hp.choice(*self._build_choice('squared_network_state')),
            hyperopt.hp.choice(*self._build_choice('activation_function')),
            hyperopt.hp.qnormal('bias_scale', 1, 1, 0.1),
            hyperopt.hp.qnormal('signal_scale', 1, 1, 0.1),
        )

    def _load_data(self, offset=0):
        super(PseudoinverseExample, self)._load_data(offset)

        self.training_outputs[1::3] = np.nan
        self.training_outputs[2::3] = np.nan

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
            bias_scale,
            signal_scale,
    ):
        self.esn = Esn(
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
        )
        self.esn.W_in *= [bias_scale, signal_scale]

        # train
        self.esn.incomplete_fit(self.training_inputs, self.training_outputs)

        # test
        predicted_outputs = [self.esn.predict(self.test_inputs[0])]
        for i in range(len(self.test_inputs)-1):
            predicted_outputs.append(self.esn.predict(predicted_outputs[i]))

        return np.array(predicted_outputs)
