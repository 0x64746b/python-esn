# coding: utf-8

"""Learn a sine signal."""


from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

from matplotlib import pyplot as plt, ticker
from matplotlib.offsetbox import AnchoredText
import numpy as np
from sklearn.metrics import mean_squared_error

from esn.preprocessing import scale


SIGNAL_LENGTH = 15000
SAMPLES_PER_PERIOD = 300  # without endpoint
NUM_TRAINING_SAMPLES = int(SIGNAL_LENGTH * 0.7)
NUM_FREQUENCY_CHANGES = int(SIGNAL_LENGTH / 200)
MAX_FREQUENCY = 5


def plot_results(
        frequencies,
        correct_outputs,
        predicted_outputs,
        mode,
        debug=()
):
    try:
        rmse = np.sqrt(mean_squared_error(correct_outputs, predicted_outputs))
    except ValueError as error:
        rmse = error

    if not debug:
        fig, data = plt.subplots()
    else:
        fig, (data, activations, output_weights) = plt.subplots(nrows=3)

    data.plot(
        frequencies,
        color='r',
        label='Input frequency'
    )
    data.plot(correct_outputs, label='Correct outputs')
    data.plot(predicted_outputs, label='Predicted outputs')
    data.xaxis.set_major_locator(
        ticker.MultipleLocator(SAMPLES_PER_PERIOD)
    )
    data.add_artist(AnchoredText('RMSE: {}'.format(rmse), loc=2))
    data.legend()
    data.set_title('Mode: {}'.format(mode))

    if debug:
        tracked_units, tracked_activations, w_out = debug
        print('W_out:', w_out.shape)

        for i, reservoir_unit in enumerate(tracked_units):
            weights = []

            for output_unit in range(w_out.shape[0]):
                weights.append(w_out[output_unit][reservoir_unit])

            activations.plot(
                tracked_activations[:,i],
                label='Unit {} (weights: {})'.format(reservoir_unit, weights))

        activations.legend()
        activations.set_title('Activations of most influential units')

        for output_unit in range(w_out.shape[0]):
            output_weights.plot(
                w_out[output_unit],
                label='Unit {}'.format(output_unit)
            )

        output_weights.legend()
        output_weights.set_title('Output weights')

    plt.tight_layout()
    plt.show()


def generate_signal(
        num_sampling_points,
        samples_per_period,
        num_frequency_changes,
        max_frequency,
):
    """
    Generate a sine signal with varying frequency.

    Inspired by https://github.com/cknd/pyESN/blob/master/freqgen.ipynb.
    """
    norm_sampling_distance = 2 * np.pi / samples_per_period

    frequencies = np.zeros(num_sampling_points)
    signal = np.zeros(num_sampling_points)

    frequency_intervals = np.sort(np.append(
        [0, num_sampling_points],
        np.random.randint(0, num_sampling_points, num_frequency_changes)
    ))

    for (start, end) in zip(frequency_intervals, frequency_intervals[1:]):
        frequencies[start:end] = np.random.randint(1, max_frequency + 1)

    sampling_point = 0
    for i in range(num_sampling_points):
        sampling_point += norm_sampling_distance * frequencies[i]
        signal[i] = np.sin(sampling_point)

    return frequencies, signal


def load_data():
    frequencies, signal = generate_signal(
        SIGNAL_LENGTH,
        SAMPLES_PER_PERIOD,
        NUM_FREQUENCY_CHANGES,
        MAX_FREQUENCY,
    )

    # scale frequencies to [-1, 1]
    frequencies = scale(frequencies)

    training_inputs = (
        frequencies[:NUM_TRAINING_SAMPLES],
        signal[:NUM_TRAINING_SAMPLES]
    )
    training_outputs = signal[1:NUM_TRAINING_SAMPLES + 1]

    # consume training data
    frequencies = np.delete(frequencies, np.s_[:NUM_TRAINING_SAMPLES])
    signal = np.delete(signal, np.s_[:NUM_TRAINING_SAMPLES])

    inputs = (frequencies[:-1], signal[:-1])
    correct_outputs = signal[1:]

    return training_inputs, training_outputs, inputs, correct_outputs


# make modules importable from the package name space.
#  import late to break cyclic import
from .generate_with_manual_feedback import Example as ManualFeedbackGenerator
from .generate_with_structural_feedback import Example as StructuralFeedbackGenerator
from .predict import Example as Predictor
from .simple import Example as UnparametrizedGenerator
