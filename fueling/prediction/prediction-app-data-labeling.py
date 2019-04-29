#!/usr/bin/env python
from dump_feature_proto import DumpFeatureProto
from generate_labels import GenerateLabels
from merge_labels import MergeLabels

if __name__ == '__main__':
    DumpFeatureProto().main()
    GenerateLabels().main()
    MergeLabels().main()
