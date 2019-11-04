#!/usr/bin/env python

""" generate todo tasks at data pipeline """

import os

from fueling.common.base_pipeline import BasePipeline
from fueling.common.storage.bos_client import BosClient


def get_todo_tasks(origin_prefix, target_prefix,
                   marker_origin='COMPLETE', marker_processed='COMPLETE'):
    """Get to be processed tasks/folders in rdd format."""
    # RDD(dir_of_file_end_with_marker_origin)
    origin_dirs = list_completed_dirs(origin_prefix, marker_origin)
    # RDD(dir_of_file_end_with_marker_processed)
    processed_dirs = (list_completed_dirs(target_prefix, marker_processed)
                      # RDD(dir_of_file_end_with_marker_processed, in orgin_prefix)
                      .map(lambda path: path.replace(target_prefix, origin_prefix, 1)))
    # RDD(dir_of_to_do_tasks)
    return origin_dirs.subtract(processed_dirs)


# Helper function
def list_completed_dirs(prefix, marker):
    """List directories that contains COMPLETE mark up files"""
    # RDD(files in prefix folders)
    return (
        # RDD(files_end_with_marker)
        BasePipeline.SPARK_CONTEXT.parallelize(BosClient().list_files(prefix, marker))
        # RDD(dirs_of_file_end_with_marker)
        .map(os.path.dirname))
