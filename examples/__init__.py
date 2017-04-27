# coding: utf-8

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import logging


def setup_logging(verbosity):
    logging.basicConfig(
        level=max(logging.DEBUG, logging.WARNING - verbosity * 10)
    )
