#!/usr/bin/env python
"""Wrapper of vehicle calibration pipeline."""

from fueling.common.base_pipeline import SequentialPipeline
from fueling.control.calibration_table.multi_job_data_distribution import MultiJobDataDistribution
from fueling.control.calibration_table.multi_job_feature_extraction import MultiJobFeatureExtraction
from fueling.control.calibration_table.multi_job_result_visualization import MultiJobResultVisualization
from fueling.control.calibration_table.multi_job_train import MultiJobTrain


if __name__ == '__main__':
    SequentialPipeline([
        MultiJobFeatureExtraction(),
        MultiJobTrain(),
        MultiJobResultVisualization(),
        MultiJobDataDistribution(),
    ]).main()
