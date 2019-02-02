#!/usr/bin/env python

from pyspark import SparkContext, SparkConf


kCurrentContext = None


def GetContext(app_name='SparkJob'):
    """Get new Spark context."""
    conf = SparkConf().setAppName(app_name)
    kCurrentContext = SparkContext(conf=conf)
    return kCurrentContext


def CurrentContext():
    """Get existing or new Spark context."""
    return kCurrentContext or GetContext()
