"""A simple demo PySpark job."""
#!/usr/bin/env python

# Standard packages
import pprint
import time

# Third-party packages
from absl import flags

# Apollo packages
from cyber_py.record import RecordReader

# Apollo-fuel packages
from fueling.common.base_pipeline import BasePipeline

flags.DEFINE_integer('sleep_time', 0, 'Time to sleep.')


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
            # PairRDD(topic, PyBagMessage)
            .keyBy(lambda msg: msg.topic)
            # PairRDD(topic, 1)
            .mapValues(lambda msg: 1)
            # PairRDD(topic, N), with unique keys.
            .reduceByKey(lambda a, b: a + b)
            # PairRDD(topic, N), just sleep.
            .map(self.sleep)
            # [(topic, N)]
            .collect())

    def run_prod(self):
        """For this demo, prod and test are the same."""
        return self.run_test()

    def sleep(self, input):
        """Dummy process to leave some time for UI show at http://localhost:4040"""
        sleep_time = self.FLAGS.get('sleep_time')
        if sleep_time:
            time.sleep(sleep_time)
        return input


if __name__ == '__main__':
    CountMsgByChannel().main()
