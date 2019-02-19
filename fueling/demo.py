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
        return (
            self.get_spark_context()
            # [record_path, ...]
            .parallelize(['/apollo/docs/demo_guide/demo_3.5.record'])
            # [message, ...]
            .flatMap(lambda record: RecordReader(record).read_messages())
            # [topic:message, ...]
            .keyBy(lambda msg: msg.topic)
            # [topic:1, ...]
            .mapValues(lambda msg: 1)
            # [topic:n, ...]
            .reduceByKey(lambda a, b: a + b)
            .map(DemoPipeline.dummy_process))

    def run_prod(self):
        """For this demo, prod and test are the same."""
        return self.run_test()

    @staticmethod
    def dummy_process(elem):
        """Dummy process to leave some time for UI show at http://localhost:4040"""
        time.sleep(60)
        return elem


if __name__ == '__main__':
    # Gather result to memory and print nicely.
    RESULT = DemoPipeline().run_test()
    pprint.PrettyPrinter().pprint(RESULT.collect())
