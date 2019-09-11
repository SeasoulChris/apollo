#!/usr/bin/env python
from absl import app

from fueling.prediction.prediction_result import PredictionResult
from fueling.prediction.evaluate_performance import PerformanceEvaluator


def performance_evaluation(argv):
    PredictionResult().__main__(argv)
    PerformanceEvaluator().__main__(argv)


if __name__ == '__main__':
    app.run(performance_evaluation)
