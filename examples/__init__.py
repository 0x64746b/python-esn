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

import hyperopt
from matplotlib import pyplot as plt, ticker
from matplotlib.offsetbox import AnchoredText
import numpy as np
from sklearn.metrics import mean_squared_error
from timeit import default_timer as timer


logger = logging.getLogger(__name__)


class EsnExample(object):

    def __init__(
            self,
            training_inputs,
            training_outputs,
            test_inputs,
            test_outputs
    ):
        self.training_inputs = training_inputs
        self.training_outputs = training_outputs

        self.test_inputs = test_inputs
        self.test_outputs = test_outputs

    def _plot_results(
            self,
            data,
            title,
            debug=None,
            periodicity=None,
            output_file=None
    ):
        plt.style.use('ggplot')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        if not debug:
            fig, main = plt.subplots()
        elif 'test_activations' in debug:
            fig, (main, extra, training_activations) = plt.subplots(nrows=3)
        else:
            fig, (main, training_activations, extra) = plt.subplots(nrows=3)

        main.set_title(title)
        data.plot(ax=main, ylim=[-1.1, 1.1]).legend(loc=1)
        if periodicity:
            main.xaxis.set_major_locator(ticker.MultipleLocator(periodicity))

        try:
            rmse = np.sqrt(mean_squared_error(
                data['Correct outputs'],
                data['Predicted outputs']
            ))
        except ValueError as error:
            rmse = error
        finally:
            error_box = AnchoredText('RMSE: {}'.format(rmse), loc=2)
            # mimic style of legend
            error_box.patch.set(
                boxstyle=main.get_legend().get_frame().get_boxstyle(),
                facecolor=main.get_legend().get_frame().get_facecolor(),
                edgecolor=main.get_legend().get_frame().get_edgecolor(),
            )
            main.add_artist(error_box)

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

        if output_file:
            logging.info('Saving plot to \'%s\'...', output_file)
            fig.set_size_inches(10.24, 7.68)
            plt.savefig(output_file, dpi=100)
        else:
            plt.show()

    def _objective(self, hyper_parameters):
        start = timer()

        # re-seed for repeatable results
        random_seed = np.random.randint(2**32)
        np.random.seed(random_seed)

        try:
            predicted_outputs = self._train(*hyper_parameters)
        except Exception as error:
            result = {'status': hyperopt.STATUS_FAIL, 'problem': str(error)}
        else:
            try:
                rmse = np.sqrt(mean_squared_error(
                    self.test_outputs,
                    predicted_outputs
                ))
            except Exception as error:
                result = {'status': hyperopt.STATUS_FAIL, 'problem': str(error)}
            else:
                result = {
                    'status': hyperopt.STATUS_OK,
                    'loss': rmse,
                    'seed': str(random_seed)
                }
        finally:
            logger.info(
                'seed: %s | sampled hyper-parameters: %s => %s  [took: %s]',
                random_seed,
                hyper_parameters,
                result['loss'] if 'loss' in result else result['problem'],
                timer() - start,
                )

            return result


def dispatch_examples():
    """The main entry point."""
    from esn.examples import mackey_glass, parameterized_sine, superposed_sine

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-v',
        '--verbose',
        dest='verbosity',
        action='count',
        default=0,
        help='Increase the log level with each use'
    )
    parser.add_argument(
        '-s',
        '--save',
        metavar='FILE',
        dest='output_file',
        help='save the generated plot to the given file instead of showing it'
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
    parameterized_sine_group = example_groups.add_parser(
        'parameterized-sine',
        help=parameterized_sine.__doc__
    )
    superposed_sine_group = example_groups.add_parser(
        'superposed-sine',
        help=superposed_sine.__doc__
    )

    #  mackey-glass examples (map to a module)
    mackey_glass_examples = mackey_glass_group.add_subparsers(
        title='examples',
        dest='example'
    )
    mackey_glass_examples.required = True

    mackey_glass_rls = mackey_glass_examples.add_parser(
        'rls',
        help=mackey_glass.rls.__doc__
    )
    mackey_glass_rls.add_argument(
        '-o',
        '--optimize',
        metavar='EXP_KEY',
        help='Optimize the hyperparameters of the example instead of running it %(default)s'
    )
    mackey_glass_rls.add_argument(
        'data_file',
        help='the file containing the data to learn'
    )

    #  parametrized sine examples (map to a module)
    parameterized_sine_examples = parameterized_sine_group.add_subparsers(
        title='examples',
        dest='example'
    )
    parameterized_sine_examples.required = True

    parameterized_sine_mlp = parameterized_sine_examples.add_parser(
        'mlp',
        help=parameterized_sine.mlp.__doc__
    )
    parameterized_sine_mlp.add_argument(
        '-o',
        '--optimize',
        metavar='EXP_KEY',
        help='Optimize the hyperparameters of the example instead of running it'
    )

    #  superposed sine examples (map to a module)
    superposed_sine_examples = superposed_sine_group.add_subparsers(
        title='examples',
        dest='example'
    )
    superposed_sine_examples.required = True

    superposed_sine_rls = superposed_sine_examples.add_parser(
        'rls',
        help=superposed_sine.rls.__doc__
    )
    superposed_sine_rls.add_argument(
        '-o',
        '--optimize',
        metavar='EXP_KEY',
        help='Optimize the hyperparameters of the example instead of running it'
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=max(logging.DEBUG, logging.WARNING - args.verbosity * 10)
    )

    # explicitly seed PRNG for comparable runs
    np.random.seed(48)

    if args.example_group == 'mackey-glass':
        example_group = mackey_glass
        data = example_group.load_data(args.data_file)
    elif args.example_group == 'parameterized-sine':
        example_group = parameterized_sine
        data = example_group.load_data()
    elif args.example_group == 'superposed-sine':
        example_group = superposed_sine
        data = example_group.load_data()

    if args.example == 'rls':
        example = example_group.RlsExample(*data)
    elif args.example == 'mlp':
        example = example_group.MlpExample(*data)

    if 'optimize' in args and args.optimize:
        example.optimize(args.optimize)
    else:
        example.run(args.output_file)
