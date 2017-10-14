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

        self.num_training_samples = 850000
        self.num_test_samples = 5000

        self.title = 'Frequency generator; LMS; {} samples'.format(
            self.num_training_samples
        )

        self.random_seed = 3013457484
        self.hyper_parameters = {
            'reservoir_size': 100,
            'spectral_radius': 1.32,
            'leaking_rate': 0.25,
            'learning_rate': 0.0005,
            'sparsity': 0.9,
            'initial_transients': 450,
            'state_noise': 0.0008439,
            'squared_network_state': True,
            'activation_function': np.tanh,
            'bias_scale': 0.06,
            'frequency_scale': -0.45,
            'signal_scale': -1.15,
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

    def _load_data(self, offset=0):
        super(LmsExample, self)._load_data(offset)

        # remove training labels to simulate incomplete data
        self.training_inputs[2::3, 1] = np.nan
        self.training_inputs[3::3, 1] = np.nan

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
        self.esn.fit(
            np.array([self.training_inputs[0]]),
            np.array([self.training_outputs[0]])
        )
        for input_date, output_date in zip(
                self.training_inputs[1:],
                self.training_outputs[1:]
        ):
            if np.isnan(input_date[1]):
                input_date[1] = prediction[0]

            if not np.isnan(output_date.item()):
                self.esn.partial_fit(
                    np.array([input_date]),
                    np.array([output_date])
                )
            else:
                # drive reservoir
                prediction = self.esn.predict(input_date)

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
