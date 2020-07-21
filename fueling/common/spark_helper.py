#!/usr/bin/env python
"""Spark related utils."""

import fueling.common.logging as logging


def cache_and_log(rdd_name, rdd, show_items=1):
    """Cache and pretty log an RDD, then return the rdd itself."""
    rdd = rdd.cache()
    count = rdd.count()
    if count == 0:
        logging.warning('{} is empty!'.format(rdd_name))
    elif show_items == 0:
        logging.info('{} has {} elements.'.format(rdd_name, count))
    elif show_items == 1:
        logging.info('{} has {} elements: [{}, ...]'.format(rdd_name, count, rdd.first()))
    else:
        logging.info('{} has {} elements: [{}, ...]'.format(
            rdd_name, count, ', '.join(rdd.take(show_items))))
    return rdd
