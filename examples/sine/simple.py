# coding: utf-8

"""Generate an unparametrized sine signal"""


from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import logging

import numpy as np

from esn import Esn
from esn.examples.sine import plot_results
from esn.preprocessing import add_noise, scale


logger = logging.getLogger(__name__)


class Example(object):

    def __init__(self):
        # generate data
        num_periods = 4000
        sampling_rate = 50  # points per period
        num_sampling_points = num_periods * sampling_rate

        training_length = int(num_sampling_points * 0.7)
        test_length = 500

        sampling_points = np.linspace(
            0,
            num_periods * 2 * np.pi,
            num=num_sampling_points
        )
        signal = scale(
            np.sin(sampling_points)
            + np.sin(2 * sampling_points)
            + np.sin(4 * sampling_points)
        ).reshape(num_sampling_points, 1)

        self.training_inputs = signal[:training_length]
        self.training_outputs = signal[1:training_length + 1]

        # consume training data
        signal = np.delete(signal, np.s_[:training_length])

        self.test_inputs = signal[:test_length]
        self.test_outputs = signal[1:test_length + 1]

    def run(self):
        predicted_outputs = self._train(
            spectral_radius=0.99,
            leaking_rate=0.33,
            bias_scale=0.2,
            signal_scale=0.2,
            state_noise=1e-7,
            input_noise=1e-7,
        )

        # debug
        for i, predicted_date in enumerate([self.test_inputs[0]] + predicted_outputs[:-1]):
            logger.debug(
                '% f -> % f (Δ % f)',
                predicted_date,
                predicted_outputs[i],
                self.test_outputs[i] - predicted_outputs[i]
            )

        plot_results(
            [],
            self.test_outputs,
            predicted_outputs,
            mode='generate simple signal'
        )

    def _train(
            self,
            spectral_radius,
            leaking_rate,
            bias_scale,
            signal_scale,
            state_noise,
            input_noise,
    ):
        self.esn = Esn(
            in_size=1,
            reservoir_size=200,
            out_size=1,
            spectral_radius=spectral_radius,
            leaking_rate=leaking_rate,
            state_noise=state_noise,
            sparsity=0.95,
            initial_transients=300,
            squared_network_state=True,
        )
        self.esn.W_in *= [bias_scale, signal_scale]

        # train
        self.esn.fit(
            add_noise(self.training_inputs, input_noise),
            self.training_outputs
        )

        # test
        predicted_outputs = [self.esn.predict(self.test_inputs[0])]
        for i in range(len(self.test_inputs)-1):
            predicted_outputs.append(self.esn.predict(predicted_outputs[i]))

        return predicted_outputs
