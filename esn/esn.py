# coding: utf-8

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)


import numpy as np


class ESN(object):

    def __init__(
            self,
            in_size,
            reservoir_size,
            out_size,
            spectral_radius,
            leaking_rate,
            washout,
            smoothing_factor,
            transfer_function=np.tanh,
    ):
        np.random.seed()

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
        self.W = np.random.rand(self.N, self.N) - 0.5
        rho_W = np.max(np.abs(np.linalg.eig(self.W)[0]))
        self.W *= spectral_radius / rho_W

        # leaking rate
        self.alpha = leaking_rate

        # activation function
        self.f = transfer_function

        # number of initial states to discard due to initial transients
        self.washout = washout

        # smoothing factor for ridge regression
        self.beta = smoothing_factor

    def fit(self, input_data, output_data):
        S = self._harvest_reservoir_states(input_data)

        # discard states contaminated by initial transients and their
        # corresponding outputs
        S = np.delete(S, np.s_[:self.washout], 0)
        output_data = np.delete(output_data, np.s_[:self.washout], 0)

        self.W_out = self._compute_output_weights(S, output_data)

    def _harvest_reservoir_states(self, u):
        """
        Drive the dynamical reservoir with the training data.

        :param u: The `K` dimensional input signal of length `n_max`.
        :return: The state collection matrix of size `n_max x (N + K + 1)`
        """
        n_max = len(u)

        # state collection matrix
        S = np.zeros((n_max, self.N + self.K + 1))

        # initial reservoir state
        self.x = np.zeros(self.N)

        for n in range(n_max):
            self.x = self._update_state(self.x, u[n])
            S[n] = np.hstack((1, u[n], self.x))

        return S

    def _update_state(self, x, u):
        """
        Step the reservoir once.

        :param x: The current reservoir state
        :param u: The current input
        :return: The next reservoir state
        """
        return (1 - self.alpha) * x + self.alpha * self.f(
            np.dot(self.W_in, np.hstack((1, u))) +
            np.dot(self.W, x)
            # TODO Add `W_fb` here
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
        self.x = self._update_state(self.x, input_date)

        return np.dot(self.W_out, np.hstack((1, input_date, self.x)))
