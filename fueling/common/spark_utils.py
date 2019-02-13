"""Spark related utils."""
#!/usr/bin/env python

from pyspark import SparkContext, SparkConf


CURRENT_CONTEXT = None

def get_context(app_name='SparkJob'):
    """Get new Spark context."""
    global CURRENT_CONTEXT
    if CURRENT_CONTEXT is None:
        conf = SparkConf().setAppName(app_name)
        CURRENT_CONTEXT = SparkContext(conf=conf)
    return CURRENT_CONTEXT

def map_key(func):
    """Map a key with func."""
    def map_key_value(key_value):
        """Actual action on the key-value input."""
        return func(key_value[0]), key_value[1]
    return map_key_value
