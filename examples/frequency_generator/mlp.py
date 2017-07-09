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
import hyperopt.mongoexp
import numpy as np

from esn import MlpEsn
from esn.activation_functions import lecun
from esn.preprocessing import add_noise
from esn.examples import EsnExample
from esn.examples.frequency_generator import (
    NUM_TRAINING_SAMPLES,
    SAMPLES_PER_PERIOD,
)


INPUT_NOISE_FACTOR = 0.03


logger = logging.getLogger(__name__)


class MlpExample(EsnExample):

    def __init__(self, *data):
        super(MlpExample, self).__init__(*data)

        self.training_inputs = np.array(list(zip(
            self.training_inputs[0],
            add_noise(self.training_inputs[1], INPUT_NOISE_FACTOR)
        )))
        self.training_outputs = np.array(self.training_outputs).reshape(
            len(self.training_outputs),
            1
        )
        self.test_inputs = np.array(list(zip(*self.test_inputs)))

    def _configure(self):
        super(MlpExample, self)._configure()

        self.title = 'Parameterized sine; MLP; {} samples'.format(
            NUM_TRAINING_SAMPLES
        )
        self.periodicity = SAMPLES_PER_PERIOD

        self.hyper_parameters = {
            'spectral_radius': 1.5,
            'leaking_rate': 0.1,
            'bias_scale': 2.6,
            'frequency_scale': 2.2,
            'signal_scale': 5.5,
        }

        self.search_space = (
            hyperopt.hp.quniform('spectral_radius', 0, 1.5, 0.01),
            hyperopt.hp.quniform('leaking_rate', 0, 1, 0.01),
            hyperopt.hp.qnormal('bias_scale', 1, 1, 0.01),
            hyperopt.hp.qnormal('frequency_scale', 1, 1, 0.01),
            hyperopt.hp.qnormal('signal_scale', 1, 1, 0.1),
        )

    def _train(
            self,
            spectral_radius,
            leaking_rate,
            bias_scale=1.0,
            frequency_scale=1.0,
            signal_scale=1.0,
            num_tracked_units=0,
    ):
        self.esn = MlpEsn(
            in_size=2,
            reservoir_size=200,
            out_size=1,
            spectral_radius=spectral_radius,
            leaking_rate=leaking_rate,
            sparsity=0.95,
            initial_transients=1000,
            squared_network_state=True,
            activation_function=lecun,
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
