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

    @staticmethod
    def run(*args):
        # format data
        num_periods = 150
        sampling_points = np.linspace(
            0,
            num_periods*2*np.pi,
            num=num_periods*50
        )
        signal = np.sin(sampling_points).reshape(num_periods*50, 1)

        training_inputs = signal[:-1]
        training_outputs = signal[1:]
        inputs = signal[:170]
        correct_outputs = signal[1:170+1]

        esn = Esn(
            in_size=1,
            reservoir_size=200,
            out_size=1,
            leaking_rate=0.3,
        )

        # train
        esn.fit(training_inputs, training_outputs)

        # test
        predicted_outputs = [esn.predict(inputs[0])]
        for i in range(len(inputs)-1):
            predicted_outputs.append(esn.predict(predicted_outputs[i]))

        #  debug
        for i, predicted_date in enumerate([inputs[0]] + predicted_outputs[:-1]):
            logger.debug(
                '% f -> % f (Δ % f)',
                predicted_date,
                predicted_outputs[i],
                correct_outputs[i] - predicted_outputs[i]
            )

        plot_results(
            [],
            correct_outputs,
            predicted_outputs,
            mode='generate simple signal'
        )
