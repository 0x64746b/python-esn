# coding: utf-8

"""
Manually feed back predicted values into an `Esn`
instead of using structural feedback.
"""

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import logging

from esn import Esn
from esn.examples.mackey_glass import plot_results


logger = logging.getLogger(__name__)


class Example(object):

    @staticmethod
    def run(training_inputs, training_outputs, test_inputs, test_outputs):
        esn = Esn(
            in_size=1,
            reservoir_size=1000,
            out_size=1,
            spectral_radius=0.75,
            leaking_rate=0.3,
            sparsity=0.95,
            initial_transients=100,
            state_noise=1e-10,
        )

        # train
        esn.fit(training_inputs, training_outputs)

        # test
        predicted_outputs = [esn.predict(test_inputs[0])]
        for i in range(len(test_inputs)-1):
            predicted_outputs.append(esn.predict(predicted_outputs[i]))

        #  debug
        for i, predicted_date in enumerate([test_inputs[0]] + predicted_outputs[:-1]):
            logger.debug(
                '% f -> % f (Δ % f)',
                predicted_date,
                predicted_outputs[i],
                test_outputs[i] - predicted_outputs[i]
            )

        plot_results(
            test_outputs,
            predicted_outputs,
            mode='generate with manual feedback'
        )
