#!/usr/bin/env python
"""Log utils."""

import os

from absl import logging as absl_log


class LogInit(object):
    INITED = False

    @classmethod
    def init(cls):
        if cls.INITED:
            return
        verbosity = os.environ.get('LOG_VERBOSITY')
        if verbosity == 'INFO':
            absl_log.set_verbosity(absl_log.INFO)
        elif verbosity == 'DEBUG':
            absl_log.set_verbosity(absl_log.DEBUG)
        elif verbosity == 'WARNING':
            absl_log.set_verbosity(absl_log.WARNING)
        elif verbosity == 'ERROR':
            absl_log.set_verbosity(absl_log.ERROR)
        elif verbosity == 'FATAL':
            absl_log.set_verbosity(absl_log.FATAL)
        absl_log.info('Absl.logging inited with verbosity ' + verbosity)


LogInit.init()
debug = absl_log.debug
info = absl_log.info
warning = absl_log.warning
error = absl_log.error
fatal = absl_log.fatal
