#!/usr/bin/env python
"""Clean records."""

import datetime
import os

from cyber_py3.record import RecordReader, RecordWriter

import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.record_utils as record_utils
from fueling.common.base_pipeline import BasePipeline


class CleanPlanningRecords(BasePipeline):
    """CleanPlanningRecords pipeline."""

    def __init__(self):
        self.metrics_prefix = 'data.pipelines.clean_planning_records.'
        self.dst_prefix = 'modules/planning/cleaned_data/ver_' \
                          + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "/"
        self.src_prefixs = [
            'small-records/2018/2018-09-11/2018-09-11-11-10-30/',
        ]

    def run_test(self):
        """Run test."""
        # RDD(record_path)
        self.dst_prefix = '/apollo/data/planning/cleaned_data/ver_' \
                          + datetime.date.today().strftime("%Y%m%d_%H%M%S") + "/"

        records = ['/apollo/modules/data/fuel/testdata/data/small.record']
        processed_records = (self.to_rdd(records)
                             # RDD(RecordMeta)
                             .map(self.process_record)
                             .count())
        logging.info('Processed {}/{} records'.format(processed_records, len(records)))

    def run(self):
        """Run prod."""
        prefixes = [
            'small-records/2018/2018-09-11/2018-09-11-11-10-30/',
        ]

        # RDD(record_path)
        records_rdd = BasePipeline.SPARK_CONTEXT.union([
            self.to_rdd(self.our_storage().list_files(prefix)).filter(record_utils.is_record_file)
            for prefix in prefixes])

        processed_records = records_rdd.map(self.process_record)
        logging.info('Processed {} records'.format(processed_records.count()))

    def process_record(self, src_record_fn):
        src_record_fn_elements = src_record_fn.split("/")
        task_id = src_record_fn_elements[-2]
        fn = src_record_fn_elements[-1]

        dst_record_fn_elements = src_record_fn_elements[:-5]
        dst_record_fn_elements.append(self.dst_prefix)
        dst_record_fn_elements.append(task_id)
        dst_record_fn_elements.append(fn)

        dst_record_fn = "/".join(dst_record_fn_elements)
        logging.info(dst_record_fn)

        msgs = []
        topic_descs = {}
        try:
            reader = RecordReader(src_record_fn)
            msgs = [msg for msg in reader.read_messages()]
            if len(msgs) == 0:
                logging.error('Failed to read any message from {}'.format(src_record_fn))
                return dst_record_fn

            for msg in msgs:
                if msg.topic not in topic_descs:
                    topic_descs[msg.topic] = (msg.data_type, reader.get_protodesc(msg.topic))
        except Exception as err:
            logging.error('Failed to read record {}: {}'.format(src_record_fn, err))
            return None

        # Write to record.
        file_utils.makedirs(os.path.dirname(dst_record_fn))
        writer = RecordWriter(0, 0)
        try:
            writer.open(dst_record_fn)
            for topic, (data_type, desc) in topic_descs.items():
                writer.write_channel(topic, data_type, desc)
            for msg in msgs:
                writer.write_message(msg.topic, msg.message, msg.timestamp)
        except Exception as e:
            logging.error('Failed to write to target file {}: {}'.format(dst_record_fn, e))
            return None
        finally:
            writer.close()
        return dst_record_fn


if __name__ == '__main__':
    CleanPlanningRecords().main()
