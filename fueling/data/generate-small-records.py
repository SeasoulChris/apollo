#!/usr/bin/env python

import fueling.common.s3_utils as s3_utils
import fueling.common.spark_utils as spark_utils


def Main():
    sc = spark_utils.GetContext()
    small_records = sc.parallelize(
        s3_utils.ListObjectKeys('apollo-platform', 'small-records/'))
    large_records = sc.parallelize(
        s3_utils.ListObjectKeys('apollo-platform', 'public-test/'))

if __name__ == '__main__':
    Main()
