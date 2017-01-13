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
        S_prime = S.T

        self.W_out = np.dot(
            np.dot(output_data, S_prime),
            np.linalg.inv(
                np.dot(S, S_prime) +
                self.beta**2 * np.eye(1 + self.K + self.N)
            )
        )

    def predict(self, input_date):
        self.x = (1 - self.alpha) * self.x + \
            self.alpha * np.tanh(
                np.dot(self.W_in, np.vstack((1, input_date))) +
                np.dot(self.W, self.x)
                # TODO Add `W_fb` here
            )
        return np.dot(self.W_out, np.vstack((1, input_date, self.x)))


if __name__ == '__main__':
    from sys import argv
    data = np.loadtxt(argv[1])

    esn = ESN(
        in_size=1,
        reservoir_size=1000,
        out_size=1,
        spectral_radius=1.25,
        leaking_rate=0.3,
        washout=100,
        smoothing_factor=0.0001
    )

    esn.fit(data[:1000], data[None, 1:1001])

    for value in range(1000, 2000):
        input_date = data[value]
        y_pred = esn.predict(input_date)
        print('{} -> {} (target: {}) | Δ: {}'.format(
            input_date,
            y_pred,
            data[value+1],
            data[value+1] - y_pred))
