#!/usr/bin/env python

from __future__ import absolute_import, print_function, unicode_literals

import setuptools


setuptools.setup(
    name='python-esn',

    version='0.1.0',

    description='Model Echo State Networks',

    url='https://github.com/0x64746b/python-esn',

    author='D.',
    author_email='dtk@gmx.de',

    license='MIT',

    keywords='ESN',

    packages=[
        'esn',
        'esn.examples',
        'esn.examples.mackey_glass',
        'esn.examples.parameterized_sine',
        'esn.examples.superposed_sine',
    ],
    package_dir={'esn.examples': 'examples'},

    install_requires=[
        'numpy>=1.12.0',
        'padasip>=1.1.0',
        'scipy>=0.18.1',
    ],
    extras_require={
        'examples': [
            'hyperopt>=0.1',
            'matplotlib>=2.0.0',
            'pandas>=0.20.2',
            'scikit-learn>=0.18.1',
        ]
    },

    entry_points={
        'console_scripts': [
            'esn_examples = esn.examples:dispatch_examples',
        ],
    }
)
