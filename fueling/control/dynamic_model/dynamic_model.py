#!/usr/bin/env python
"""Wrapper of dynamic model pipeline."""

from fueling.common.base_pipeline import SequentialPipeline
from fueling.control.dynamic_model.feature_extraction.sample_set import SampleSet
from fueling.control.dynamic_model.feature_extraction.uniform_set import UniformSet
from fueling.control.dynamic_model.dynamic_model_training import DynamicModelTraining
from fueling.control.dynamic_model.dynamic_model_data_visualization import DynamicModelDatasetDistribution

if __name__ == '__main__':
    SequentialPipeline([
        # the total pipeline of dynamic model training does not work properly, please divide
        # feature extration and model training
        # SampleSet(),
        # UniformSet(),
        DynamicModelTraining(),
    ]).main()
