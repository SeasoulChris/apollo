import os

import fueling.common.colored_glog as glog


def list_end_files(origin_dir):
    """
    List all end files recursively under the specified dir.  
    This is for testing used by run_test() which simulates the production behavior
    """
    # glog.info('origin_dir: {}'.format(origin_dir))
    end_files = list()
    for (dirpath, _, filenames) in os.walk(origin_dir):
        end_files.extend([os.path.join(dirpath, file_name)
                         for file_name in filenames])

    # glog.info('end_files: {}'.format(end_files))
    return end_files


def list_completed_dirs(prefix, list_func, marker):
    """List directories that contains COMPLETE mark up files"""
    # RDD(files in prefix folders)
    return (list_func(prefix) \
        # RDD(files_end_with_marker)
        .filter(lambda path: path.endswith(marker)) \
        # RDD(dirs_of_file_end_with_marker)       
        .map(os.path.dirname))


def get_todo_tasks(origin_prefix, target_prefix, list_func, marker_origin, marker_processed):
    """Get to be processed files in rdd format."""
    # RDD(dir_of_file_end_with_marker_origin)
    origin_dirs = list_completed_dirs(origin_prefix, list_func, marker_origin)

#     glog.info('origin_dir: {}'.format(origin_dirs.collect()))

    # RDD(dir_of_file_end_with_marker_processed)
    processed_dirs = (list_completed_dirs(target_prefix, list_func, marker_processed)
    # RDD(dir_of_file_end_with_marker_processed, in orgin_prefix)
            .map(lambda path: path.replace(target_prefix, origin_prefix, 1)))

    glog.info('processed_dirs: {}'.format(origin_dirs.collect()))
    # RDD(dir_of_to_do_files)
    return origin_dirs.subtract(processed_dirs)
