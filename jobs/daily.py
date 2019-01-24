#!/usr/bin/env python

from pyspark import SparkContext, SparkConf
import urllib2

conf = SparkConf().setAppName('Test')
sc = SparkContext(conf=conf)

sc.parallelize([1, 3, 5, 7, 9]) \
  .map(lambda num: num + 1) \
  .saveAsTextFile('file:///mnt/nfs/test/spark-output')
