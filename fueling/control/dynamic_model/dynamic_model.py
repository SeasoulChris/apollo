#!/usr/bin/env python
"""Wrapper of dynamic model pipeline."""

from fueling.common.base_pipeline import SequentialPipeline
from fueling.control.dynamic_model.dynamic_model_data_visualization \
    import DynamicModelDatasetDistribution
from fueling.control.dynamic_model.dynamic_model_training import DynamicModelTraining
from fueling.control.dynamic_model.flag import flags
from fueling.control.dynamic_model.feature_extraction.sample_set import SampleSet
from fueling.control.dynamic_model.feature_extraction.uniform_set import UniformSet
import fueling.common.logging as logging


if __name__ == '__main__':
    SequentialPipeline([
        SampleSet(),
        UniformSet(),
        DynamicModelDatasetDistribution(),
        DynamicModelTraining(),
    ]).main()
    if flags.FLAGS.is_backward:
        logging.info('Backward dynamic modeling finished.')
    else:
        logging.info('Forward dynamic modeling finished.')
