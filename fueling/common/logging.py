#!/usr/bin/env python
"""Log utils."""

import os

from absl.logging import *


class LogInit(object):
    INITED = False

    @classmethod
    def init(cls):
        if cls.INITED:
            return
        verbosity = os.environ.get('LOG_VERBOSITY')
        if verbosity == 'INFO':
            set_verbosity(INFO)
        elif verbosity == 'DEBUG':
            set_verbosity(DEBUG)
        elif verbosity == 'WARNING':
            set_verbosity(WARNING)
        elif verbosity == 'ERROR':
            set_verbosity(ERROR)
        elif verbosity == 'FATAL':
            set_verbosity(FATAL)
        info('Absl.logging inited with verbosity ' + verbosity)


LogInit.init()
