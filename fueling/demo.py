"""A simple demo PySpark job."""
#!/usr/bin/env python

import pprint
import time

import pyspark_utils.helper as spark_helper

from cyber_py.record import RecordReader


def dummy_process(elem):
    """Dummy process to leave some time for UI show at http://localhost:4040"""
    time.sleep(60)
    return elem

if __name__ == '__main__':
    # Spark cascade style programming.
    RESULT = (
        spark_helper.get_context('Test')
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
        .map(dummy_process))

    # Gather result to memory and print nicely.
    pprint.PrettyPrinter().pprint(RESULT.collect())
