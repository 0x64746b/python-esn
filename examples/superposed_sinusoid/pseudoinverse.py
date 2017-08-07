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

    def _configure(self):
        super(PseudoinverseExample, self)._configure()

        self.num_training_samples = 10000
        self.num_test_samples = 500

        self.title = 'Superposed sine; Pseudoinverse; {} samples'.format(
            self.num_training_samples
        )

        self.hyper_parameters = {
            'reservoir_size': 3000,
            'spectral_radius': 1.11,
            'leaking_rate': 0.75,
            'sparsity': 0.95,
            'initial_transients': 1000,
            'state_noise': 1e-5,
            'squared_network_state': True,
            'activation_function': lecun,
            'bias_scale': -0.4,
            'signal_scale': 1.2,
        }

        self.search_space = (
            hyperopt.hp.quniform('reservoir_size', 3000, 3001, 1000),
            hyperopt.hp.quniform('spectral_radius', 0.01, 2, 0.01),
            hyperopt.hp.quniform('leaking_rate', 0.01, 1, 0.01),
            hyperopt.hp.quniform('sparsity', 0.01, 0.99, 0.01),
            hyperopt.hp.quniform('initial_transients', 100, 1001, 100),
            hyperopt.hp.quniform('state_noise', 1e-7, 1e-2, 1e-7),
            hyperopt.hp.choice(*self._build_choice('squared_network_state')),
            hyperopt.hp.choice(*self._build_choice('activation_function')),
            hyperopt.hp.qnormal('bias_scale', 1, 1, 0.1),
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
        self.esn.fit(self.training_inputs, self.training_outputs)

        # test
        predicted_outputs = [self.esn.predict(self.test_inputs[0])]
        for i in range(len(self.test_inputs)-1):
            predicted_outputs.append(self.esn.predict(predicted_outputs[i]))

        return np.array(predicted_outputs)
