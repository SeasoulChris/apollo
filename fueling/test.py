#!/usr/bin/env python

from pyspark import SparkContext, SparkConf

conf = SparkConf().setAppName('Test')
evens = SparkContext(conf=conf) \
  .parallelize([1, 3, 5, 7, 9]) \
  .map(lambda num: num + 1) \
  .collect()

print evens
