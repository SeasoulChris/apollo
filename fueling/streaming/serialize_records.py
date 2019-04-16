#!/usr/bin/env python

"""This script deserialize records to lists of messages and meta data when they arrive"""

from absl import flags
from pyspark.streaming import StreamingContext
import colored_glog as glog

from fueling.common.base_pipeline import BasePipeline
import fueling.common.s3_utils as s3_utils
import fueling.streaming.serialize_utils as serialize_utils
from fueling.streaming.streaming_listener import DriverStreamingListener
import fueling.streaming.streaming_utils as streaming_utils

class DeserializeRecordsPipeline(BasePipeline):
    """Deserialize records pipeline."""

    def __init__(self):
        BasePipeline.__init__(self, 'deserialize-records')
        # For working around "process_stream", which is not able to accept packed parameters
        self._root_dir = '/apollo'

    def run_test(self):
        """Run test."""
        self._root_dir = '/apollo'
        self.run()
        glog.info('Serialization: All Done. TEST')

    def run_prod(self):
        """Run prod."""
        self._root_dir = s3_utils.S3_MOUNT_PATH
        self.run()
        glog.info('Serialization: All Done. PROD')

    def run(self):
        """Run streaming process"""
        stream_context = StreamingContext(self.context(), 30)
        stream_context.addStreamingListener(DriverStreamingListener())

        record_path = streaming_utils.get_streaming_records(self._root_dir)
        glog.info('Streaming monitors at {}'.format(record_path))

        partitions = flags.FLAGS.executors or 4
        glog.info('partition number: {}'.format(partitions))

        records = stream_context.textFileStream(record_path).repartition(partitions)
        records.pprint()

        records.foreachRDD(self.process_stream)

        stream_context.start()
        stream_context.awaitTermination()

    def process_stream(self, stime, records_rdd):
        """Executor running"""
        glog.info('stream processing time: {}'.format(stime))
        glog.info('rdd partitions: {}'.format(records_rdd.getNumPartitions()))
        records_rdd.map(lambda record: serialize_utils.parse_record(record, self._root_dir)).count()

if __name__ == '__main__':
    DeserializeRecordsPipeline().run_test()
