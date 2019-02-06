#!/usr/bin/env python

from cyber_py.record import RecordReader

import fueling.common.spark_utils as spark_utils


print(spark_utils.GetContext('Test')
    # [record_path, ...]
    .parallelize(['/apollo/docs/demo_guide/demo_3.5.record'])
    # [message, ...]
    .flatMap(lambda record: RecordReader(record).read_messages())
    # [topic:message, ...]
    .keyBy(lambda msg: msg.topic)
    # [topic:1, ...]
    .mapValues(lambda msg: 1)
    # [topic:n, ...]
    .reduceByKey(lambda a, b: a + b)
    # Trigger action and get result to local memory.
    .collect())
