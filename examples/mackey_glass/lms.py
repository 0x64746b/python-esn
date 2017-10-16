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
from . import MackeyGlassExample


logger = logging.getLogger(__name__)


class LmsExample(MackeyGlassExample):

    def __init__(self, *args, **kwargs):
        super(LmsExample, self).__init__(*args, **kwargs)

        self.num_training_samples = 150000
        self.num_test_samples = 500

        self.title = 'Mackey-Glass; LMS; {} samples'.format(
            self.num_training_samples
        )

        self.random_seed = 3459763109
        self.hyper_parameters = {
            'reservoir_size': 100,
            'spectral_radius': 1.4,
            'leaking_rate': 0.35,
            'learning_rate': 0.00044,
            'sparsity': 0.25,
            'initial_transients': 200,
            'state_noise': 0.004009,
            'squared_network_state': True,
            'activation_function': lecun,
            'bias_scale': 0.58,
            'signal_scale': 5.4,
        }

        self.search_space = (
            hyperopt.hp.quniform('reservoir_size', 100, 101, 10),
            hyperopt.hp.quniform('spectral_radius', 0.01, 2, 0.01),
            hyperopt.hp.quniform('leaking_rate', 0.01, 1, 0.01),
            hyperopt.hp.qloguniform('learning_rate', np.log(0.00001), np.log(0.1), 0.00001),
            hyperopt.hp.quniform('sparsity', 0.01, 0.99, 0.01),
            hyperopt.hp.quniform('initial_transients', 50, 301, 50),
            hyperopt.hp.quniform('state_noise', 1e-7, 1e-2, 1e-7),
            hyperopt.hp.choice(*self._build_choice('squared_network_state')),
            hyperopt.hp.choice(*self._build_choice('activation_function')),
            hyperopt.hp.qnormal('bias_scale', 1, 1, 0.01),
            hyperopt.hp.qnormal('signal_scale', 1, 1, 0.1),
        )

    def _load_data(self, offset=0):
        super(LmsExample, self)._load_data(offset)

        # remove training labels to simulate incomplete data
        self.training_inputs[2::3] = np.nan
        self.training_inputs[3::3] = np.nan

        self.training_outputs[1::3] = np.nan
        self.training_outputs[2::3] = np.nan

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
        #print(' * fitting first data pair')
        self.esn.fit(
            np.array([self.training_inputs[0]]),
            np.array([self.training_outputs[0]])
        )
        for input_date, output_date in zip(
                self.training_inputs[1:],
                self.training_outputs[1:]
        ):
            if np.isnan(input_date):
                #print(' ** input is nan, using last prediction')
                input_date = prediction

            if not np.isnan(output_date.item()):
                #print(' * fitting pair')
                self.esn.partial_fit(
                    np.array([input_date]),
                    np.array([output_date])
                )
            else:
                # drive reservoir
                prediction = self.esn.predict(input_date)
                #print(' * generated prediction:', prediction)

        # test
        predicted_outputs = [self.esn.predict(self.test_inputs[0])]
        for i in range(len(self.test_inputs) - 1):
            predicted_outputs.append(self.esn.predict(predicted_outputs[i]))

        return np.array(predicted_outputs)
