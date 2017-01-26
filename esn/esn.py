# coding: utf-8

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)


import numpy as np
from scipy import sparse

from esn.preprocessing import add_noise

from . import transfer_functions


class ESN(object):

    def __init__(
            self,
            in_size,
            reservoir_size,
            out_size,
            spectral_radius,
            leaking_rate,
            washout,
            sparsity=0,
            output_feedback=False,
            teacher_noise=0,
            activation_function=np.tanh,
            output_activation_function=transfer_functions.identity,
            ridge_regression=0,
    ):
        # dimension of input signal
        self.K = in_size

        # number of reservoir units
        self.N = reservoir_size

        # dimension of output signal
        self.L = out_size

        # input weight matrix
        # - centered around zero
        # - of intermediate size in order to avoid the flat error surfaces
        #   near the origin and far from the origin
        self.W_in = np.random.rand(self.N, self.K + 1) - 0.5

        # reservoir weight matrix
        self.W = sparse.rand(self.N, self.N, density=1-sparsity, format='csc')
        self.W[self.W != 0] -= 0.5
        rho_W = np.abs(sparse.linalg.eigs(
            self.W,
            k=1,
            return_eigenvectors=False
        )[0])
        self.W *= spectral_radius / rho_W

        # output feedback matrix
        if output_feedback:
            self.W_fb = np.random.rand(self.N, self.L) - 0.5
        else:
            self.W_fb = np.zeros((self.N, self.L))

        # amount of noise added when forcing teacher outputs
        self.mu = teacher_noise

        # leaking rate
        self.alpha = leaking_rate

        # transfer function of the neurons in the reservoir
        self.f = activation_function

        # output activation function
        self.g = output_activation_function

        # number of initial states to discard due to initial transients
        self.washout = washout

        # smoothing factor for ridge regression
        self.beta = ridge_regression

        # initial reservoir state
        self.x = np.zeros(self.N)

        # initial output
        self.y = np.zeros(self.L)

    def fit(self, input_data, output_data):
        teacher_outputs = add_noise(output_data, self.mu)

        S = self._harvest_reservoir_states(input_data, teacher_outputs)

        # discard states contaminated by initial transients and their
        # corresponding outputs
        S = np.delete(S, np.s_[:self.washout], 0)
        output_data = np.delete(output_data, np.s_[:self.washout], 0)

        self.W_out = self._compute_output_weights(S, output_data)

    def _harvest_reservoir_states(self, u, y):
        """
        Drive the dynamical reservoir with the training data.

        :param u: The `K` dimensional input signal of length `n_max`.
        :param y: The `L` dimensional output signal.
        :return: The state collection matrix of size `n_max x (N + K + 1)`
        """
        n_max = len(u)

        # state collection matrix
        S = np.zeros((n_max, self.N + self.K + 1))

        for n in range(n_max):
            self.x = self._update_state(u[n], self.x, self.y)
            self.y = y[n]
            S[n] = np.hstack((1, u[n], self.x))

        return S

    def _update_state(self, u, x, y):
        """
        Step the reservoir once.

        :param u: The current input
        :param x: The current reservoir state
        :param y: The previous output
        :return: The next reservoir state
        """
        return (1 - self.alpha) * x + self.alpha * self.f(
            np.dot(self.W_in, np.hstack((1, u))) +
            self.W.dot(x) +
            np.dot(self.W_fb, y)
        )

    def _compute_output_weights(self, S, D):
        """
        Compute the output weights.

        They are the linear regression weights of the teacher outputs on the
        reservoir states.

        :param S: The state collection matrix of size `n_max x (N + K + 1)`
        :param D: The teacher output collection matrix of size `n_max x L`
        :return: The output weights of size `L x (N + K + 1)`
        """
        R = np.dot(S.T, S)
        P = np.dot(S.T, D)

        # Ridge regression
        return np.dot(
            np.linalg.inv(R + self.beta**2 * np.identity(1 + self.K + self.N)),
            P
        ).T

    def predict(self, input_date):
        self.x = self._update_state(input_date, self.x, self.y)
        self.y = self.g(np.dot(self.W_out, np.hstack((1, input_date, self.x))))

        return self.y
