#!/usr/bin/env python
from fueling.prediction.dump_feature_proto import DumpFeatureProto
from fueling.prediction.generate_labels import GenerateLabels
from fueling.prediction.merge_labels import MergeLabels

if __name__ == '__main__':
    DumpFeatureProto().main()
    GenerateLabels().main()
    MergeLabels().main()
