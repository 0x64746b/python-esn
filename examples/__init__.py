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
import pandas as pd
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

        self._configure()

    def _configure(self):
        self.title = ''
        self.periodicity = None

        self.random_seed = 42
        self.hyper_parameters = {}

        self.search_space = ()

    def run(self, output_file):
        np.random.seed(self.random_seed)

        predicted_outputs = self._train(**self.hyper_parameters)

        # debug
        self._log_debug(predicted_outputs)

        self._plot_results(
            data=self._get_plotting_data(predicted_outputs),
            output_file=output_file
        )

    def _log_debug(self, predicted_outputs):
        for i, predicted_date in enumerate(np.concatenate((
                [self.test_inputs[0]],
                predicted_outputs[:-1])
        )):
            logger.debug(
                '% f -> % f (Δ % f)',
                predicted_date,
                predicted_outputs[i],
                self.test_outputs[i] - predicted_outputs[i]
            )

    def _get_plotting_data(self, predicted_outputs):
        return {
            'Correct outputs': self.test_outputs.flatten(),
            'Predicted outputs': predicted_outputs.flatten(),
        }

    def _plot_results(self, data, debug=None, output_file=None):
        plt.style.use('ggplot')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        if not debug:
            fig, main = plt.subplots()
        elif 'test_activations' in debug:
            fig, (main, extra, training_activations) = plt.subplots(nrows=3)
        else:
            fig, (main, training_activations, extra) = plt.subplots(nrows=3)

        main.set_title(self.title)
        pd.DataFrame(data).plot(ax=main, ylim=[-1.1, 1.1]).legend(loc=1)
        if self.periodicity:
            main.xaxis.set_major_locator(
                ticker.MultipleLocator(self.periodicity)
            )

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
                if self.periodicity:
                    extra.xaxis.set_major_locator(ticker.MultipleLocator(
                        self.periodicity
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

    def optimize(self, exp_key):
        trials = hyperopt.mongoexp.MongoTrials(
            'mongo://localhost:27017/python_esn_trials/jobs',
            exp_key=exp_key,
        )

        best = hyperopt.fmin(
            self._objective,
            space=self.search_space,
            algo=hyperopt.tpe.suggest,
            max_evals=500,
            trials=trials,
        )

        logger.info('Best parameter combination: %s', best)

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
    from esn.examples import (
        mackey_glass,
        frequency_generator,
        superposed_sinusoid,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-v',
        '--verbose',
        dest='verbosity',
        action='count',
        default=0,
        help='Increase the log level with each use'
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '-s',
        '--save',
        metavar='FILE',
        dest='output_file',
        help='Run the experiment but save the generated plot to the given file'
             ' instead of showing it'
    )
    mode_group.add_argument(
        '-o',
        '--optimize',
        metavar='EXP_KEY',
        help='Optimize the hyper-parameters of the example'
             ' instead of running it'
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
    frequency_generator_group = example_groups.add_parser(
        'frequency-generator',
        help=frequency_generator.__doc__
    )
    superposed_sinusoid_group = example_groups.add_parser(
        'superposed-sinusoid',
        help=superposed_sinusoid.__doc__
    )

    #  mackey-glass examples (map to a module)
    mackey_glass_group.add_argument(
        '-n',
        '--network-type',
        metavar='TYPE',
        choices=['pinv', 'lms', 'rls'],
        default='pinv',
        help='The type of network to train (default: %(default)s)'
    )
    mackey_glass_group.add_argument(
        'data_file',
        help='the file containing the data to learn'
    )

    #  frequency generator examples (map to a module)
    frequency_generator_group.add_argument(
        '-n',
        '--network-type',
        metavar='TYPE',
        choices=['pinv', 'mlp'],
        default='pinv',
        help='The type of network to train (default: %(default)s)'
    )

    #  superposed sinusoid examples (map to a module)
    superposed_sinusoid_group.add_argument(
        '-n',
        '--network-type',
        metavar='TYPE',
        choices=['pinv', 'rls'],
        default='pinv',
        help='The type of network to train (default: %(default)s)'
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=max(logging.DEBUG, logging.WARNING - args.verbosity * 10)
    )

    # explicitly seed PRNG for reproducible data generation
    np.random.seed(42)

    if args.example_group == 'mackey-glass':
        example_group = mackey_glass
        data = example_group.load_data(args.data_file)
    elif args.example_group == 'frequency-generator':
        example_group = frequency_generator
        data = example_group.load_data()
    elif args.example_group == 'superposed-sinusoid':
        example_group = superposed_sinusoid
        data = example_group.load_data()

    if args.network_type == 'pinv':
        example = example_group.PseudoinverseExample(*data)
    elif args.network_type == 'lms':
        example = example_group.LmsExample(*data)
    elif args.network_type == 'rls':
        example = example_group.RlsExample(*data)
    elif args.network_type == 'mlp':
        example = example_group.MlpExample(*data)

    if args.optimize:
        example.optimize(args.optimize)
    else:
        example.run(args.output_file)
