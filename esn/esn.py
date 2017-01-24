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
        # state collection matrix
        S = np.zeros((self.N + self.K + 1, len(input_data)))

        # harvest_reservoir_states
        #  initial reservoir state
        self.x = np.zeros((self.N, 1))
        for i in range(len(input_data)):
            u = input_data[i]
            self.x = (1 - self.alpha) * self.x + \
                self.alpha * np.tanh(
                    np.dot(self.W_in, np.vstack((1, u))) +
                    np.dot(self.W, self.x)
                    # TODO Add `W_fb` here
                )
            S[:, i] = np.vstack((1, u, self.x))[:, 0]

        # discard states contaminated by initial transients and their
        # corresponding outputs
        S = np.delete(S, np.s_[:self.washout], 1)
        output_data = np.delete(output_data, np.s_[:self.washout], 1)

        # compute output weights
        R = np.dot(S, S.T)
        P = np.dot(output_data, S.T)

        self.W_out = np.dot(
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
