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

    packages=['esn', 'esn.examples'],
    package_dir={'esn.examples': 'examples'},

    install_requires=[
        'numpy>=1.12.0',
        'scipy>=0.18.1',
    ],
    extras_require={
        'examples': [
            'matplotlib>=2.0.0',
            'scikit-learn>=0.18.1',
        ]
    },

    entry_points={
        'console_scripts': [
            'esn_sine = esn.examples.sine:main',
            'esn_mackey_glass = esn.examples.mackey_glass:main',
        ],
    }
)
