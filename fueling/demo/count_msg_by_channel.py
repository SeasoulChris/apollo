"""
A simple demo PySpark job to stat messages by channel.

Prerequisite:
    cd /apollo/docs/demo_guide
    python rosbag_helper.py demo_3.5.record

Run with:
    ./tools/submit-job-to-k8s.py --entrypoint=fueling/demo/count_msg_by_channel.py
"""
#!/usr/bin/env python

# Standard packages
import pprint

# Apollo packages
from cyber_py.record import RecordReader

# Apollo-fuel packages
from fueling.common.base_pipeline import BasePipeline
import fueling.common.logging as logging


class CountMsgByChannel(BasePipeline):
    """Demo pipeline."""

    def __init__(self):
        BasePipeline.__init__(self, 'demo')

    def run_test(self):
        # Spark cascade style programming.
        pprint.PrettyPrinter().pprint(
            # RDD(record_path)
            self.to_rdd(['/apollo/docs/demo_guide/demo_3.5.record'])
            # RDD(PyBagMessage)
            .flatMap(lambda record: RecordReader(record).read_messages())
            # PairRDD(topic, 1)
            .map(lambda msg: (msg.topic, 1))
            # PairRDD(topic, N), with unique keys.
            .reduceByKey(lambda a, b: a + b)
            # [(topic, N)]
            .collect())
        logging.info('Pipeline finished!')

    def run_prod(self):
        """For this demo, prod and test are the same."""
        return self.run_test()


if __name__ == '__main__':
    CountMsgByChannel().main()
