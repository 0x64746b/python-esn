# coding: utf-8

from __future__ import absolute_import, print_function, unicode_literals


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
            smoothing_factor
    ):
        np.random.seed()

        # dimension of input signal
        self.K = in_size

        # number of reservoir units
        self.N = reservoir_size

        # dimension of output signal
        self.L = out_size

        # input weight matrix
        self.W_in = (np.random.rand(self.N, self.K + 1) - 0.5) * 1

        # reservoir weight matrix
        self.W = np.random.rand(self.N, self.N) - 0.5
        rho_W = np.max(np.abs(np.linalg.eig(self.W)[0]))
        self.W *= spectral_radius / rho_W

        # leaking rate
        self.alpha = leaking_rate

        # number of initial states to discard due to initial transients
        self.washout = washout

        # smoothing factor for ridge regression
        self.beta = smoothing_factor

    def fit(self, input_data, output_data):
        S = self._harvest_reservoir_states(input_data)

        # discard states contaminated by initial transients and their
        # corresponding outputs
        S = np.delete(S, np.s_[:self.washout], 1)
        output_data = np.delete(output_data, np.s_[:self.washout], 1)

        self.W_out = self._compute_output_weights(S, output_data)

    def _harvest_reservoir_states(self, u):
        """
        Drive the dynamical reservoir with the training data.

        :param u: The `K` dimensional input signal of length `n_max`.
        :return: The state collection matrix of size `(N + K + 1) x n_max`
        """
        # state collection matrix
        S = np.zeros((self.N + self.K + 1, len(u)))

        # initial reservoir state
        self.x = np.zeros((self.N, 1))

        for n in range(len(u)):
            self.x = (1 - self.alpha) * self.x + self.alpha * np.tanh(
                 np.dot(self.W_in, np.vstack((1, u[n]))) +
                 np.dot(self.W, self.x)
                 # TODO Add `W_fb` here
            )
            S[:, n] = np.vstack((1, u[n], self.x))[:, 0]

        return S

    def _compute_output_weights(self, S, D):
        """
        Compute the output weights.

        They are the linear regression weights of the teacher outputs on the
        reservoir states.

        :param S: The state collection matrix of size `(N + K + 1) x n_max`
        :param D: The teacher output collection matrix of size `L x n_max`
        :return: The output weights of size `L x (N + K + 1)`
        """
        R = np.dot(S, S.T)
        P = np.dot(D, S.T)

        return np.dot(
            P,
            np.linalg.inv(R + self.beta**2 * np.identity(1 + self.K + self.N))
        )

    def predict(self, input_date):
        self.x = (1 - self.alpha) * self.x + \
            self.alpha * np.tanh(
                np.dot(self.W_in, np.vstack((1, input_date))) +
                np.dot(self.W, self.x)
                # TODO Add `W_fb` here
            )
        return np.dot(self.W_out, np.vstack((1, input_date, self.x)))
