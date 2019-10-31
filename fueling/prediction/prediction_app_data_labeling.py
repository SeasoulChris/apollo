#!/usr/bin/env python
"""Wrapper of prediction jobs."""

from fueling.common.base_pipeline import SequentialPipeline
from fueling.prediction.dump_feature_proto import DumpFeatureProto
from fueling.prediction.generate_labels import GenerateLabels
from fueling.prediction.merge_labels import MergeLabels


if __name__ == '__main__':
    SequentialPipeline([
        DumpFeatureProto(),
        GenerateLabels(),
        MergeLabels(),
    ]).main()
