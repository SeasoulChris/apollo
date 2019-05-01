#!/usr/bin/env python
from absl import app

from fueling.prediction.dump_feature_proto import DumpFeatureProto
from fueling.prediction.generate_labels import GenerateLabels
from fueling.prediction.merge_labels import MergeLabels

def data_labeling(argv):
    DumpFeatureProto().__main__(argv)
    GenerateLabels().__main__(argv)
    MergeLabels().__main__(argv)

if __name__ == '__main__':
    app.run(data_labeling)
