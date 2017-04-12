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


logger = logging.getLogger(__name__)


class Example(object):

    def __init__(self):
        # format data
        num_periods = 150
        sampling_points = np.linspace(
            0,
            num_periods*2*np.pi,
            num=num_periods*50
        )
        signal = np.sin(sampling_points).reshape(num_periods*50, 1)

        self.training_inputs = signal[:-1]
        self.training_outputs = signal[1:]
        self.test_inputs = signal[:170]
        self.test_outputs = signal[1:170 + 1]


    def run(self):
        predicted_outputs = self._train()

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

    def _train(self):
        self.esn = Esn(
            in_size=1,
            reservoir_size=200,
            out_size=1,
            leaking_rate=0.3,
        )

        # train
        self.esn.fit(self.training_inputs, self.training_outputs)

        # test
        predicted_outputs = [self.esn.predict(self.test_inputs[0])]
        for i in range(len(self.test_inputs)-1):
            predicted_outputs.append(self.esn.predict(predicted_outputs[i]))

        return predicted_outputs
