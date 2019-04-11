""" get to do tasks"""
import glob
import os

import fueling.common.s3_utils as s3_utils

def list_end_files(origin_dir):
    """
    List all end files recursively under the specified dir.
    This is for testing used by run_test() which simulates the production behavior
    """
    end_files = list()
    for (dirpath, _, filenames) in os.walk(origin_dir):
        end_files.extend([os.path.join(dirpath, file_name)
                          for file_name in filenames])
    return end_files


def list_completed_dirs(prefix, list_func, marker):
    """List directories that contains COMPLETE mark up files"""
    # RDD(files in prefix folders)
    return (list_func(prefix) \
            # RDD(files_end_with_marker)
            .filter(lambda path: path.endswith(marker)) \
            # RDD(dirs_of_file_end_with_marker)
            .map(os.path.dirname))

def get_todo_tasks(origin_prefix, target_prefix, list_func, 
                   marker_origin='COMPLETE', marker_processed='COMPLETE'):
    """Get to be processed files in rdd format."""
    # RDD(dir_of_file_end_with_marker_origin)
    origin_dirs = list_completed_dirs(origin_prefix, list_func, marker_origin)
    # RDD(dir_of_file_end_with_marker_processed)
    processed_dirs = (list_completed_dirs(target_prefix, list_func, marker_processed)
                      # RDD(dir_of_file_end_with_marker_processed, in orgin_prefix)
                      .map(lambda path: path.replace(target_prefix, origin_prefix, 1)))
    # RDD(dir_of_to_do_files)
    return origin_dirs.subtract(processed_dirs)

def get_todo_tasks_prod(origin_prefix, target_prefix, root_dir, bucket, MARKER):
    list_func = (lambda path: s3_utils.list_files(bucket, path))
    # RDD(record_dir)
    todo_task_dirs = (get_todo_tasks(
        origin_prefix, target_prefix, list_func, '/COMPLETE', '/' + MARKER))
    todo_tasks = (
        # RDD(record_dir)
        todo_task_dirs
        # RDD(abs_record_dir)
        .map(lambda record_dir: os.path.join(root_dir, record_dir))
        # PairRDD(record_dir, record_dir)
        .keyBy(lambda record_dir: record_dir)
        # PairRDD(record_dir, record_files)
        .flatMapValues(lambda path: glob.glob(os.path.join(path, '*record*'))))
    return todo_tasks
    
