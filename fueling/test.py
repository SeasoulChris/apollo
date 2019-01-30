#!/usr/bin/env python

from pyspark import SparkContext, SparkConf

from cyber_py.record import RecordReader

conf = SparkConf().setAppName('Test')
sc = SparkContext(conf=conf)

res = sc.parallelize(\
  ['/mnt/bos/small-records/2018-04-23/mkz9/2018-04-23-15-24-10/2018-04-23-15-52-13_28.record']) \
  .flatMap(lambda record: RecordReader(record).read_messages()) \
  .map(lambda msg: (msg.channel_name, 1)) \
  .reduceByKey(lambda v1, v2: v1 + v2) \
  .collect()

print(res)
