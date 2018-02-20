# coding: utf-8

"""Learn the Mackey-Glass equation."""


from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import numpy as np

from esn.examples import EsnExample
from esn.preprocessing import scale


class MackeyGlassExample(EsnExample):

    def __init__(self, data_file):
        self._data_file = data_file
        super(MackeyGlassExample, self).__init__()

    def _load_data(self, offset=0):
        data = np.loadtxt(self._data_file)

        # shift to a fresh set of data
        discarded = offset * (self.num_training_samples + self.num_test_samples)
        data = np.delete(data, np.s_[:discarded])

        # scale data to stretch to [-1, 1]
        data = scale(data)

        self.training_inputs = data[:self.num_training_samples].reshape(
            self.num_training_samples,
            1  # in_size
        )
        self.training_outputs = data[1:self.num_training_samples+1].reshape(
            self.num_training_samples,
            1  # out_size
        ).copy()

        # consume training data
        data = np.delete(data, np.s_[:self.num_training_samples])

        self.test_inputs = data[:self.num_test_samples].reshape(
            self.num_test_samples,
            1
        )
        self.test_outputs = data[1:self.num_test_samples + 1]


# make modules importable from the package name space.
#  import late to break cyclic import
from .pseudoinverse import PseudoinverseExample
from .lms import LmsExample
from .rls import RlsExample
from .mlp import MlpExample
