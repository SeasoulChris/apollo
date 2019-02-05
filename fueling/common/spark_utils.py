#!/usr/bin/env python

from pyspark import SparkContext, SparkConf


kCurrentContext = None

def GetContext(app_name='SparkJob'):
    """Get new Spark context."""
    global kCurrentContext
    if kCurrentContext is None:
        conf = SparkConf().setAppName(app_name)
        kCurrentContext = SparkContext(conf=conf)
    return kCurrentContext

def MapKey(func):
    """Map a key with func."""
    def MapKeyValue(key_value):
        return func(key_value[0]), key_value[1]
    return MapKeyValue
