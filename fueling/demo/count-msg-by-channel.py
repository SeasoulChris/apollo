"""A simple demo PySpark job."""
#!/usr/bin/env python

# Standard packages
import pprint
import time

# Third-party packages
from absl import flags
import colored_glog as glog

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
            # RDD(record_path), just sleep.
            .map(self.sleep)
            # RDD(PyBagMessage)
            .flatMap(lambda record: RecordReader(record).read_messages())
            # PairRDD(topic, 1)
            .map(lambda msg: (msg.topic, 1))
            # PairRDD(topic, N), with unique keys.
            .reduceByKey(lambda a, b: a + b)
            # [(topic, N)]
            .collect())

    def run_prod(self):
        """For this demo, prod and test are the same."""
        return self.run_test()

    def sleep(self, input):
        """Dummy process to run longer so you can access Spark UI or check logs."""
        for i in range(self.FLAGS.get('sleep_time', 0)):
            glog.info('Tick {}'.format(i))
            time.sleep(1)
        return input


if __name__ == '__main__':
    CountMsgByChannel().main()
