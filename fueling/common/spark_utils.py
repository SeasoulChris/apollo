#!/usr/bin/env python

from pyspark import SparkContext, SparkConf

def GetContext(app_name='SparkJob'):
    """Get Spark context."""
    conf = SparkConf().setAppName(app_name)
    return SparkContext(conf=conf)
