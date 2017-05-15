# coding: utf-8

"""Demo the usage of the `python-esn` library"""

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import argparse
import logging

import numpy as np

from esn.examples import mackey_glass, sine


def dispatch_examples():
    """The main entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-v',
        '--verbose',
        dest='verbosity',
        action='count',
        default=0,
        help='Increase the log level with each use'
    )

    # example groups (map to a package)
    example_groups = parser.add_subparsers(
        title='example groups',
        dest='example_group'
    )
    example_groups.required = True

    mackey_glass_group = example_groups.add_parser(
        'mackey-glass',
        help=mackey_glass.__doc__
    )
    sine_group = example_groups.add_parser(
        'sine',
        help=sine.__doc__
    )

    #  mackey-glass examples (map to a module)
    mackey_glass_examples = mackey_glass_group.add_subparsers(
        title='examples',
        dest='example'
    )
    mackey_glass_examples.required = True

    mackey_glass_predict = mackey_glass_examples.add_parser(
        'predict',
        help=mackey_glass.predict.__doc__
    )
    mackey_glass_predict.add_argument(
        'data_file',
        help='the file containing the data to learn'
    )

    mackey_glass_generate = mackey_glass_examples.add_parser(
        'generate',
        help=mackey_glass.generate_with_structural_feedback.__doc__
    )
    mackey_glass_generate.add_argument(
        '-m',
        '--with-manual-feedback',
        action='store_true',
        help=mackey_glass.generate_with_manual_feedback.__doc__
    )
    mackey_glass_generate.add_argument(
        'data_file',
        help='the file containing the data to learn'
    )

    #  sine examples (map to a module)
    sine_examples = sine_group.add_subparsers(
        title='examples',
        dest='example'
    )
    sine_examples.required = True

    sine_simple = sine_examples.add_parser(
        'simple',
        help=sine.simple.__doc__
    )
    sine_simple.add_argument(
        '-o',
        '--optimize',
        metavar='EXP_KEY',
        help='Optimize the hyperparameters of the example instead of running it'
    )

    sine_examples.add_parser(
        'predict',
        help=sine.predict.__doc__
    )

    sine_generate = sine_examples.add_parser(
        'generate',
        help=sine.generate_with_structural_feedback.__doc__
    )
    sine_generate.add_argument(
        '-m',
        '--with-manual-feedback',
        action='store_true',
        help=sine.generate_with_manual_feedback.__doc__
    )
    sine_generate.add_argument(
        '-o',
        '--optimize',
        metavar='EXP_KEY',
        help='Optimize the hyperparameters of the example instead of running it'
    )

    args = parser.parse_args()

    setup_logging(args.verbosity)

    # explicitly seed PRNG for comparable runs
    np.random.seed(48)

    if args.example_group == 'mackey-glass':
        example_group = mackey_glass
        data = example_group.load_data(args.data_file)
    elif args.example_group == 'sine':
        example_group = sine
        data = example_group.load_data()

    if args.example == 'generate':
        if not args.with_manual_feedback:
            example = example_group.StructuralFeedbackGenerator(*data)
        else:
            example = example_group.ManualFeedbackGenerator(*data)
    elif args.example == 'predict':
        example = example_group.Predictor(*data)
    elif args.example == 'simple':
        example = example_group.UnparametrizedGenerator()

    if 'optimize' in args and args.optimize:
        example.optimize(args.optimize)
    else:
        example.run()


def setup_logging(verbosity):
    logging.basicConfig(
        level=max(logging.DEBUG, logging.WARNING - verbosity * 10)
    )
