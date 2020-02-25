"""
A simple demo PySpark job to stat messages by channel.

Run with:
    bazel run //fueling/demo:count_msg_by_channel
"""
#!/usr/bin/env python

# Standard packages
import pprint

# Apollo packages
from cyber_py3.record import RecordReader

# Apollo-fuel packages
from fueling.common.base_pipeline_v2 import BasePipelineV2
import fueling.common.file_utils as file_utils


class CountMsgByChannel(BasePipelineV2):
    """Demo pipeline."""

    def run(self):
        # Spark cascade style programming.
        pprint.PrettyPrinter().pprint(
            # RDD(record_path)
            self.to_rdd(['fueling/demo/testdata/small.record'])
            # RDD(record_abs_path)
            .map(file_utils.fuel_path)
            # RDD(PyBagMessage)
            .flatMap(lambda record: RecordReader(record).read_messages())
            # PairRDD(topic, 1)
            .map(lambda msg: (msg.topic, 1))
            # PairRDD(topic, N), with unique keys.
            .reduceByKey(lambda a, b: a + b)
            # [(topic, N)]
            .collect())


if __name__ == '__main__':
    CountMsgByChannel().main()
