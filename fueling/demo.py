"""A simple demo PySpark job."""
#!/usr/bin/env python
import pprint
import time

from cyber_py.record import RecordReader

from fueling.common.base_pipeline import BasePipeline


class DemoPipeline(BasePipeline):
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
            .map(DemoPipeline.dummy_process)
            # [(topic, N)]
            .collect())

    def run_prod(self):
        """For this demo, prod and test are the same."""
        return self.run_test()

    def run_grpc(self):
        """For this demo, grpc and test are the same."""
        return self.run_test()

    @staticmethod
    def dummy_process(elem):
        """Dummy process to leave some time for UI show at http://localhost:4040"""
        time.sleep(20)
        return elem


if __name__ == '__main__':
    DemoPipeline().main()
