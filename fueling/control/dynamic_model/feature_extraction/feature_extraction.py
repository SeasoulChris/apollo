#!/usr/bin/env python
"""Wrapper of dynamic model pipeline."""

from fueling.common.base_pipeline import SequentialPipeline
from fueling.control.dynamic_model.feature_extraction.sample_set import SampleSet
from fueling.control.dynamic_model.feature_extraction.uniform_set import UniformSet

if __name__ == '__main__':
    SequentialPipeline([
        SampleSet(),
        UniformSet(),
    ]).main()
