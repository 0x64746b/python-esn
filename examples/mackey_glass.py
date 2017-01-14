#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, print_function, unicode_literals


from sys import argv

import numpy as np
from sklearn.metrics import mean_squared_error

from esn import ESN


NUM_TRAINING_SAMPLES = 1000
NUM_PREDICTION_SAMPLES = 1000


def main(training_inputs, training_outputs, inputs, correct_outputs):
    predicted_outputs = []

    esn = ESN(
        in_size=1,
        reservoir_size=1000,
        out_size=1,
        spectral_radius=1.25,
        leaking_rate=0.3,
        washout=100,
        smoothing_factor=0.0001
    )

    esn.fit(training_inputs, training_outputs)

    for i in range(len(inputs)):
        input_date = inputs[i]
        predicted_output = esn.predict(input_date)
        predicted_output = predicted_output[0][0]

        print(
            '{:0< 18} -> {:0< 18} (Δ {:< 18})'.format(
                input_date,
                predicted_output,
                correct_outputs[i] - predicted_output
            )
        )

        predicted_outputs.append(predicted_output)

    mse = mean_squared_error(correct_outputs, predicted_outputs)
    print('\n MSE: {}'.format(mse))


if __name__ == '__main__':
    data = np.loadtxt(argv[1])

    training_inputs = data[:NUM_TRAINING_SAMPLES]
    training_outputs = data[None, 1:NUM_TRAINING_SAMPLES+1]

    np.delete(data, np.s_[:NUM_TRAINING_SAMPLES])

    inputs = data[:NUM_PREDICTION_SAMPLES]
    correct_outputs = data[1:NUM_PREDICTION_SAMPLES+1]

    main(training_inputs, training_outputs, inputs, correct_outputs)
