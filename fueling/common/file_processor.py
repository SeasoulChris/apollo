#!/usr/bin/env python
"""File processor."""

import sys

import google.protobuf.json_format as json_format


class FileProcessor(object):
    # The processor should be stateless.
    def process_file_to_proto(self, filepath):
        """Process a file to a proto."""
        raise Exception('FileProcessor::process_file_to_proto not implemented!')

    def process_file_to_json(self, filepath):
        """You should implement this function, which """
        return json_format.MessageToJson(self.process_file_to_proto(filepath))


# An example processor in record_profiling.py:
#
# class RecordProfiling(FileProcessor):
#     def process_file_to_proto(self, filepath):
#         return result_pb
# if __name__ == '__main__':
#     print(RecordProfiling().process_file_to_json(sys.argv[1]))


# In Apollo Fuel, you may call it like
#   RDD(records).map(RecordProfiling().process_file_to_proto)
#
# In Dreamland, you may call it like
#   python record_profiling.py "/path/to/record"
