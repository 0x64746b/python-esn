#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, print_function, unicode_literals


from sys import argv

import numpy as np

from esn import ESN


if __name__ == '__main__':

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
