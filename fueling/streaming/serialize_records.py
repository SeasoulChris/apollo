#!/usr/bin/env python

"""This script deserialize records to lists of messages and meta data when they arrive"""
import os

from pyspark.streaming import StreamingContext

from fueling.common.base_pipeline import BasePipeline
import fueling.streaming.serialize_utils as serialize_utils

class DeserializeRecordsPipeline(BasePipeline):
    """Deserialize records pipeline."""

    def __init__(self):
        BasePipeline.__init__(self, 'deserialize-records')

    def run_test(self):
        """Run test."""
        spark_context = self.get_spark_context()
        stream_context = StreamingContext(spark_context, 30)
        root_dir = '/apollo'
        streaming_dir = 'modules/data/fuel/testdata/streaming'
        records = stream_context.textFileStream(
            os.path.join(root_dir, os.path.join(streaming_dir, 'records')))

        partitions = int(os.environ.get('APOLLO_EXECUTORS', 4))
        records.repartition(partitions)
        records.foreachRDD(
            lambda rdd: rdd.foreachPartition(
                lambda partition: serialize_utils.parse_records(
                    partition, root_dir, streaming_dir)))

        stream_context.start()
        stream_context.awaitTermination()

    def run_prod(self):
        """Run prod."""
        pass

if __name__ == '__main__':
    DeserializeRecordsPipeline().run_test()
