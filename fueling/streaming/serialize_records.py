#!/usr/bin/env python

"""This script deserialize records to lists of messages and meta data when they arrive"""

import os

from pyspark.streaming import StreamingContext

from fueling.common.base_pipeline import BasePipeline
from fueling.streaming.streaming_listener import DriverStreamingListener
import fueling.common.s3_utils as s3_utils
import fueling.streaming.serialize_utils as serialize_utils
import fueling.streaming.streaming_utils as streaming_utils

class DeserializeRecordsPipeline(BasePipeline):
    """Deserialize records pipeline."""

    def __init__(self):
        BasePipeline.__init__(self, 'deserialize-records')

    def run_test(self):
        """Run test."""
        self.run('/apollo')

    def run_prod(self):
        """Run prod."""
        self.run(s3_utils.S3_MOUNT_PATH)

    def run(self, root_dir):
        """Run streaming process"""
        spark_context = self.get_spark_context()
        stream_context = StreamingContext(spark_context, 30)
        stream_context.addStreamingListener(DriverStreamingListener())
        records = stream_context.textFileStream(streaming_utils.get_streaming_records(root_dir))

        partitions = int(os.environ.get('APOLLO_EXECUTORS', 4))
        records.repartition(partitions)
        records.foreachRDD(
            lambda rdd: rdd.foreachPartition(
                lambda partition: serialize_utils.parse_records(
                    partition, root_dir)))

        stream_context.start()
        stream_context.awaitTermination()

if __name__ == '__main__':
    DeserializeRecordsPipeline().run_test()
