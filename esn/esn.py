# coding: utf-8

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import numpy as np
import padasip as pa
from scipy import sparse
from scipy.sparse import linalg

from . import activation_functions
from .preprocessing import add_noise


class Esn(object):
    """
    Model an Echo State Network.

    Use the pseudoinverse of the extended reservoir states to compute the output
    weights.
    """

    BIAS = np.array([1])

    def __init__(
            self,
            in_size,
            reservoir_size,
            out_size,
            spectral_radius=0.99,
            leaking_rate=1,
            sparsity=0,
            initial_transients=0,
            state_noise=0,
            squared_network_state=False,
            activation_function=np.tanh,
            output_activation_function=(
                activation_functions.identity,
                activation_functions.identity
            ),
            output_feedback=False,
            teacher_noise=0,
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
        self.W_in = np.random.rand(self.N, self.BIAS.size + self.K) - 0.5

        # reservoir weight matrix
        self.W = sparse.rand(self.N, self.N, density=1-sparsity, format='csc')
        self.W[self.W != 0] -= 0.5
        rho_W = np.abs(sparse.linalg.eigs(
            self.W,
            k=1,
            return_eigenvectors=False
        )[0])
        self.W *= spectral_radius / rho_W

        # the untrained output weights
        self.W_out = None

        # output feedback matrix
        if output_feedback:
            self.W_fb = np.random.rand(self.N, self.L) - 0.5
        else:
            self.W_fb = np.zeros((self.N, self.L))

        # amount of noise added when forcing teacher outputs
        self.mu = teacher_noise

        # leaking rate
        self.alpha = leaking_rate

        # number of states to discard due to initial transients
        self.washout = initial_transients

        # state noise factor
        self.nu = state_noise

        # use the squared network state as additional nonlinear transformations
        self.nonlinear_augmentation = squared_network_state

        # transfer function of the neurons in the reservoir
        self.f = activation_function

        # output activation function
        #  make sure to scale the target outputs to within the domain of the
        #  inverse output function
        self.g = output_activation_function[0]
        self.g_inv = output_activation_function[1]

        # initial reservoir state
        self.x = np.zeros(self.N)

        # initial output
        self.y = np.zeros(self.L)

        # the number of most influential reservoir units to track
        #  the actual tracked number can be lower if a unit is the most
        #  influential one for multiple outputs.
        self.num_tracked_units = 0
        self.tracked_units = {}

    @classmethod
    def _prepend_bias(cls, input_data, sequence=False):
        bias = cls.BIAS if not sequence else np.broadcast_to(
            cls.BIAS,
            (input_data.shape[0], cls.BIAS.shape[0])
        )
        return np.hstack((bias, input_data))

    def fit(self, input_data, output_data):
        u = self._prepend_bias(input_data, sequence=True)
        y = add_noise(output_data, self.mu)

        S = self._harvest_reservoir_states(u, y)

        # discard states contaminated by initial transients and their
        # corresponding outputs
        S = np.delete(S, np.s_[:self.washout], 0)
        D = np.delete(output_data, np.s_[:self.washout], 0)

        self.W_out = self._compute_output_weights(S, D)

        if self.num_tracked_units:
            self.tracked_units = self.track_most_influential_units(S)

    def _harvest_reservoir_states(self, u, y):
        """
        Drive the dynamical reservoir with the training data.

        :param u: The `1 + K` dimensional input signal of length `n_max`.
        :param y: The `L` dimensional output signal.
        :return: The state collection matrix of size `n_max x (1 + K + N)`
        """
        n_max = len(u)

        # state collection matrix
        S = np.zeros((n_max, self.BIAS.size + self.K + self.N))

        for n in range(n_max):
            self.x = self._update_state(u[n], self.x, self.y, self.nu)
            self.y = y[n]
            S[n] = np.hstack((u[n], self.x))

        if self.nonlinear_augmentation:
            S = np.hstack((S, S[:, 1:] ** 2))

        return S

    def _update_state(self, u, x, y, nu=0):
        """
        Step the reservoir once.

        :param u: The current input
        :param x: The current reservoir state
        :param y: The previous output
        :param nu: The amount of state noise
        :return: The next reservoir state
        """
        return (1 - self.alpha) * x + self.alpha * (
            self.f(
                self.W_in.dot(u)
                + self.W.dot(x)
                + self.W_fb.dot(y)
                + nu * (np.random.rand(self.N) - 0.5)
            )
        )

    def _compute_output_weights(self, S, D):
        """
        Compute the output weights.

        They are the linear regression weights of the teacher outputs on the
        reservoir states.

        Note that `S` (and therefore the returned weights) may have `K + N`
        additional columns if the states have been squared to add nonlinearity.

        :param S: The state collection matrix of size `n_max x (1 + K + N)`
        :param D: The teacher output collection matrix of size `n_max x L`
        :return: The output weights of size `L x (1 + K + N)`
        """
        return np.dot(
            np.linalg.pinv(S),
            self.g_inv(D)
        ).T

    def track_most_influential_units(self, S):
        # flat indices of highest weights
        flat_indices = np.argpartition(
            np.abs(self.W_out).flatten(),
            -self.num_tracked_units)[-self.num_tracked_units:]

        tracked_units = np.vstack(np.unravel_index(
            flat_indices,
            self.W_out.shape
        ))[1]

        return {unit: S[:, unit] for unit in tracked_units}

    def predict(self, input_date):
        u = self._prepend_bias(input_date)

        self.x = self._update_state(u, self.x, self.y)
        z = np.hstack((u, self.x))

        if self.nonlinear_augmentation:
            z = np.hstack((z, z[1:] ** 2))

        self.y = self.g(np.dot(self.W_out, z))

        return self.y


class WienerHopfEsn(Esn):
    """
    Model an Echo State Network.

    Invoke the Wiener-Hopf solution to compute the output weights.
    """

    def __init__(self, ridge_regression=0, *args, **kwargs):
        super(WienerHopfEsn, self).__init__(*args, **kwargs)

        # smoothing factor for ridge regression
        self.beta = ridge_regression

    def _compute_output_weights(self, S, D):
        # See super method for docstring
        R = np.dot(S.T, S)
        P = np.dot(S.T, self.g_inv(D))

        # Ridge regression
        return np.dot(
            np.linalg.inv(
                R
                + self.beta**2 * np.identity(R.shape[1])
            ),
            P
        ).T


class LmsEsn(Esn):
    """
    Model an Echo State Network.

    Update the output weights online through an adaptive LMS filter.
    """

    def fit(self, input_data, output_data):
        u = self._prepend_bias(input_data, sequence=True)
        y_teach = add_noise(output_data, self.mu)

        state_size = self.BIAS.size + self.K + self.N
        if self.nonlinear_augmentation:
            state_size += self.K + self.N

        n_max = len(u)

        # TODO: one per output neuron?
        lms = pa.filters.FilterLMS(
            state_size,  # TODO: Think about output feedback
            mu=0.001,  # TODO: Make configurable (as self.lambda?)
            w=str('zeros'),
        )

        if self.num_tracked_units:
            tracked_states = np.zeros((n_max, state_size))

        for n in range(n_max):
            self.x = self._update_state(u[n], self.x, self.y, self.nu)
            self.y = y_teach[n]

            if n > self.washout:
                v = np.hstack((u[n], self.x))
                if self.nonlinear_augmentation:
                    v = np.hstack((v, v[1:]**2))

                # y = rls.predict(v)
                lms.adapt(self.g_inv(y_teach[n]), v)

                if self.num_tracked_units:
                    tracked_states[n] = v

        self.W_out = lms.w.reshape((self.L, state_size))  # TODO: Think about multiple outputs

        if self.num_tracked_units:
            self.tracked_units = self.track_most_influential_units(tracked_states[self.washout:])
