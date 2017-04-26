# coding: utf-8

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np
from sklearn.metrics import mean_squared_error

from esn.preprocessing import scale


NUM_TRAINING_SAMPLES = 2000
NUM_PREDICTION_SAMPLES = 1000


def plot_results(reference, predicted, mode):
    try:
        rmse = np.sqrt(mean_squared_error(reference, predicted))
    except ValueError as error:
        rmse = error.message

    plt.plot(reference, label='Reference')
    plt.plot(predicted, label='Predicted')
    plt.gca().add_artist(AnchoredText('RMSE: {}'.format(rmse), loc=2))
    plt.gca().set_title('Mode: {}'.format(mode))
    plt.legend()
    plt.show()


def load_data(file_name):
    data = np.loadtxt(file_name)

    # scale data to stretch to [-1, 1]
    data = scale(data)

    training_inputs = data[:NUM_TRAINING_SAMPLES].reshape(
        NUM_TRAINING_SAMPLES,
        1  # in_size
    )
    training_outputs = data[1:NUM_TRAINING_SAMPLES+1].reshape(
        NUM_TRAINING_SAMPLES,
        1  # out_size
    )

    # consume training data
    data = np.delete(data, np.s_[:NUM_TRAINING_SAMPLES])

    inputs = data[:NUM_PREDICTION_SAMPLES].reshape(NUM_PREDICTION_SAMPLES, 1)
    correct_outputs = data[1:NUM_PREDICTION_SAMPLES+1]

    return training_inputs, training_outputs, inputs, correct_outputs
