#!/usr/bin/env python
"""Wrapper of prediction jobs."""

from fueling.common.base_pipeline import SequentialPipeline
from fueling.prediction.prediction_result import PredictionResult
from fueling.prediction.evaluate_performance import PerformanceEvaluator


if __name__ == '__main__':
    SequentialPipeline([
        PredictionResult(),
        PerformanceEvaluator(),
    ]).main()
