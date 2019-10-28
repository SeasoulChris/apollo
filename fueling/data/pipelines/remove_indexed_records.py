#!/usr/bin/env python
"""
Remove indexed records by prefix.
Usage:

./tools/submit-job-to-k8s.py --main=fueling/data/pipelines/remove_indexed_records.py \
    --flags="--prefix_of_indexed_records_to_remove=/mnt/bos/..."
"""

from absl import flags

from fueling.common.base_pipeline import BasePipeline
from fueling.common.mongo_utils import Mongo
import fueling.common.logging as logging


flags.DEFINE_string('prefix_of_indexed_records_to_remove', None,
                    'The prefix of records to remove from MongoDB.')


class RemoveIndexedRecords(BasePipeline):
    """RemoveIndexedRecords pipeline."""

    def __init__(self):
        BasePipeline.__init__(self, 'remove-indexed-records')

    def run_prod(self):
        """Run prod."""
        prefix = self.FLAGS.get('prefix_of_indexed_records_to_remove')
        if not prefix:
            logging.error('Please specify --prefix_of_indexed_records_to_remove.')
            return
        col = Mongo().record_collection()
        op_filter = {'path': {'$regex': '^' + prefix}}
        col.delete_many(op_filter)


if __name__ == '__main__':
    RemoveIndexedRecords().main()
