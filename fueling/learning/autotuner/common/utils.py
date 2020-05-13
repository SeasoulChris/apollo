
import time

import fueling.common.logging as logging


def run_with_retry(max_retries, func, *params):
    """A wrapper to run a given function with certain amount of retries"""
    for trial in range(max_retries):
        try:
            if trial > 0:
                logging.info(f'Retry function {func.__name__}. Retry count {trial} ...')
            result = func(*params)
            return result
        except Exception as error:
            if trial == (max_retries - 1):
                raise error
            else:
                logging.error(error)
                time.sleep(10)
