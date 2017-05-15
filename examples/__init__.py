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

from matplotlib import pyplot as plt, ticker
from matplotlib.offsetbox import AnchoredText
import numpy as np
from sklearn.metrics import mean_squared_error


def dispatch_examples():
    """The main entry point."""
    from esn.examples import mackey_glass, sine

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


def plot_results(data, mode, debug=None, periodicity=None):
    plt.style.use('ggplot')

    if not debug:
        fig, main = plt.subplots()
    elif 'test_activations' in debug:
        fig, (main, extra, training_activations) = plt.subplots(nrows=3)
    else:
        fig, (main, training_activations, extra) = plt.subplots(nrows=3)

    main.set_title('Mode: {}'.format(mode))
    data.plot(ax=main)
    if periodicity:
        main.xaxis.set_major_locator(ticker.MultipleLocator(periodicity))

    try:
        rmse = np.sqrt(mean_squared_error(
            data['correct outputs'],
            data['predicted outputs']
        ))
    except ValueError as error:
        rmse = error
    finally:
        main.add_artist(AnchoredText('RMSE: {}'.format(rmse), loc=2))

    if debug:
        training_activations.set_title(
            'Activations of most influential units during training'
        )
        training_activations.plot(
            np.array(list(debug['training_activations'].values())).T
        )
        training_activations.legend(
            [
                'Unit {} (weights: {})'.format(unit, debug['w_out'][:, unit])
                for unit in debug['training_activations']
            ]
        )

        if 'test_activations' in debug:
            extra.set_title('Activations during prediction')
            extra.plot(np.array(list(debug['test_activations'].values())).T)
            extra.legend(
                ['Unit {}'.format(unit) for unit in debug['test_activations']]
            )
            if periodicity:
                extra.xaxis.set_major_locator(ticker.MultipleLocator(
                    periodicity
                ))
        else:
            extra.set_title('Output weights')
            extra.plot(debug['w_out'].T)
            extra.legend([
                'Unit {}'.format(unit)
                for unit in range(debug['w_out'].shape[0])
            ])

    plt.tight_layout()
    plt.show()
