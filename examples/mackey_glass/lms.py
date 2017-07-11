# coding: utf-8

"""Train a least mean squares filter to compute the output weights."""

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import logging

import hyperopt
import numpy as np

from esn import LmsEsn
from esn.activation_functions import lecun
from esn.examples import EsnExample
from esn.examples.mackey_glass import NUM_TRAINING_SAMPLES


logger = logging.getLogger(__name__)


class LmsExample(EsnExample):

    def _configure(self):
        super(LmsExample, self)._configure()

        self.title = 'Mackey-Glass; LMS; {} samples'.format(
            NUM_TRAINING_SAMPLES
        )

        self.random_seed = 1839385064
        self.hyper_parameters = {
            'reservoir_size': 500,
            'spectral_radius': 1.11,
            'leaking_rate': 0.75,
            'learning_rate': 0.00002,
            'sparsity': 0.95,
            'initial_transients': 300,
            'state_noise': 0.004,
            'squared_network_state': True,
            'activation_function': lecun,
            'bias_scale': -0.4,
            'signal_scale': 1.2,
        }

        self.search_space = (
            hyperopt.hp.quniform('reservoir_size', 3000, 3001, 1000),
            hyperopt.hp.quniform('spectral_radius', 0.01, 2, 0.01),
            hyperopt.hp.quniform('leaking_rate', 0.01, 1, 0.01),
            hyperopt.hp.qloguniform('learning_rate', np.log(0.00001), np.log(0.1), 0.00001),
            hyperopt.hp.quniform('sparsity', 0.01, 0.99, 0.01),
            hyperopt.hp.quniform('initial_transients', 100, 1001, 100),
            hyperopt.hp.quniform('state_noise', 1e-7, 1e-2, 1e-7),
            hyperopt.hp.choice('squared_network_state', [False, True]),
            hyperopt.hp.choice('activation_function', [np.tanh, lecun]),
            hyperopt.hp.qnormal('bias_scale', 1, 1, 0.01),
            hyperopt.hp.qnormal('signal_scale', 1, 1, 0.1),
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
