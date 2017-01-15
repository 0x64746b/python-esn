#!/usr/bin/env python
# coding: utf-8

"""Learn the Mackey-Glass equation."""


from __future__ import absolute_import, print_function, unicode_literals

import argparse

import numpy as np
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
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

    plt.plot(correct_outputs, label='correct outputs')
    plt.plot(predicted_outputs, label='predicted outputs')
    plt.gca().add_artist(AnchoredText('MSE: {}'.format(mse), loc=2))
    plt.gca().set_title('Mode: Prediction')
    plt.legend()
    plt.show()


def load_data(file_name):
    data = np.loadtxt(file_name)

    training_inputs = data[:NUM_TRAINING_SAMPLES]
    training_outputs = data[None, 1:NUM_TRAINING_SAMPLES+1]

    np.delete(data, np.s_[:NUM_TRAINING_SAMPLES])

    inputs = data[:NUM_PREDICTION_SAMPLES]
    correct_outputs = data[1:NUM_PREDICTION_SAMPLES+1]

    return training_inputs, training_outputs, inputs, correct_outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'data_file',
        help='the file containing the data to learn'
    )

    args = parser.parse_args()

    data = load_data(args.data_file)
    main(*data)
