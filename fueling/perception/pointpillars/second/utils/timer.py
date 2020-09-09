from contextlib import contextmanager
import time

import fueling.common.logging as logging


@contextmanager
def simple_timer(name=''):
    t = time.time()
    yield
    logging.info("{} exec time: {}".format(name, time.time() - t))
