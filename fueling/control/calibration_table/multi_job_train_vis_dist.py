#!/usr/bin/env python
"""Wrapper of vehicle calibration table training, visualization and distribution pipelines."""

from fueling.common.base_pipeline import SequentialPipeline
from fueling.control.calibration_table.multi_job_data_distribution import MultiJobDataDistribution
from fueling.control.calibration_table.multi_job_result_visualization import MultiJobResultVisualization
from fueling.control.calibration_table.multi_job_train import MultiJobTrain


if __name__ == '__main__':
    SequentialPipeline([
        MultiJobTrain(),
        MultiJobResultVisualization(),
        MultiJobDataDistribution(),
    ]).main()
