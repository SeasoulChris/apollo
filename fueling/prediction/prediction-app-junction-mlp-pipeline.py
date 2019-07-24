#!/usr/bin/env python

"""Running training job for junction-mlp-pipeline"""

import colored_glog as glog

import learning_algorithms.prediction.models.junction_mlp_model.junction_mlp_pipeline \
       as junction_mlp

from fueling.common.base_pipeline import BasePipeline


class JunctionMlpPipeline(BasePipeline):
    """Junction mlp pipeline."""

    def __init__(self):
        BasePipeline.__init__(self, 'prediction junction mlp pipeline')

    def run_test(self):
        """Run test."""
        glog.info('Running Test for mlp pipeline')
        source_path = '/apollo/modules/data/fuel/testdata/prediction/junciton_small.h5'
        save_dir_path = '/apollo/modules/data/fuel/testdata/prediction'
        self.to_add([(source_path, save_dir_path)]).foreach(junction_mlp.do_training)
        glog.info('Done with running Test')

    def run_prod(self):
        """Run prod."""
        glog.info('Running Production for mlp pipeline')
        # TODO(All), add real data for training in Prod
        glog.info('Done with running Production')

if __name__ == '__main__':
    JunctionMlpPipeline().main()

