#!/usr/bin/env python

"""This script deserialize records to lists of messages and meta data when they arrive"""

import os

from pyspark.streaming import StreamingContext

from fueling.common.base_pipeline import BasePipeline
import fueling.common.logging as logging
import fueling.common.storage.bos_client as bos_client
import fueling.streaming.serialize_utils as serialize_utils
import fueling.streaming.streaming_utils as streaming_utils


class DeserializeRecordsPipeline(BasePipeline):
    """Deserialize records pipeline."""

    def __init__(self):
        BasePipeline.__init__(self)
        # For working around "process_stream", which is not able to accept packed parameters
        self._root_dir = '/apollo'

    def run_test(self):
        """Run test."""
        self._root_dir = '/apollo'
        self.run()
        logging.info('Serialization: All Done. TEST')

    def run_prod(self):
        """Run prod."""
        self._root_dir = bos_client.BOS_MOUNT_PATH
        self.run()
        logging.info('Serialization: All Done. PROD')

    def run(self):
        """Run streaming process"""
        stream_context = StreamingContext(self.context(), 30)

        record_path = streaming_utils.get_streaming_records(self._root_dir)
        logging.info('Streaming monitors at {}'.format(record_path))

        partitions = int(os.environ.get('APOLLO_EXECUTORS', 4))
        logging.info('partition number: {}'.format(partitions))

        records = stream_context.textFileStream(record_path).repartition(partitions)
        records.pprint()

        records.foreachRDD(self.process_stream)

        stream_context.start()
        stream_context.awaitTermination()

    def process_stream(self, stime, records_rdd):
        """Executor running"""
        logging.info('stream processing time: {}'.format(stime))
        logging.info('rdd partitions: {}'.format(records_rdd.getNumPartitions()))
        records_rdd.map(lambda record: serialize_utils.parse_record(record, self._root_dir)).count()


if __name__ == '__main__':
    DeserializeRecordsPipeline().main()
