#!/usr/bin/env python
"""
This is a module to extraction logsim scenarios from records
based on disengage info
"""
import os

from absl import logging

from cyber_py.record import RecordReader

from fueling.common.base_pipeline import BasePipeline
import fueling.common.record_utils as record_utils


def list_end_files(target_dir):
    """
    List all end files recursively under the specified dir.
    This is for testing used by run_test() which simulates the production behavior
    """
    logging.info('target_dir: {}'.format(target_dir))
    end_files = list()
    for (dirpath, _, filenames) in os.walk(target_dir):
        end_files.extend([os.path.join(dirpath, file_name) for file_name in filenames])

    logging.info('end_files: {}'.format(end_files))
    return end_files


def list_completed_dirs(prefix, list_func):
    """List directories that contains COMPLETE mark up files"""
    return list_func(prefix) \
        .filter(lambda path: path.endswith('/COMPLETE')) \
        .map(os.path.dirname)


def get_todo_tasks(original_prefix, target_prefix, list_func):
    """Get todo tasks in rdd format."""
    original_dirs = list_completed_dirs(original_prefix, list_func)
    processed_dirs = list_completed_dirs(target_prefix, list_func).map(
        lambda path: path.replace(target_prefix, original_prefix, 1))
    return original_dirs.subtract(processed_dirs)


def execute_task(task):
    """Execute task by task"""
    dest_dir, source_dir = task
    logging.info('Executing task with src_dir: {}, dst_dir: {}'.format(source_dir, dest_dir))

    record_file = next((x for x in os.listdir(source_dir) if record_utils.is_record_file(x)), None)
    if record_file is None:
        logging.warning('No records file found in {}'.format(source_dir))
        return
    map_dir = '/mnt/bos/modules/map/data/san_mateo'
    message = next((x for x in RecordReader(os.path.join(source_dir, record_file)).read_messages()
                    if x.topic == record_utils.ROUTING_RESPONSE_HISTORY_CHANNEL), None)
    if message is not None:
        if record_utils.message_to_proto(message).map_version.startswith('sunnyvale'):
            map_dir = '/mnt/bos/modules/map/data/sunnyvale'

    # Invoke logsim_generator binary
    logging.info("Start to extract logsim scenarios for {} and map {}".format(source_dir, map_dir))
    simulation_path = '/apollo/modules/simulation'
    return_code = os.system(
        'cd {} && ./bin/logsim_generator_executable --alsologtostderr '
        '--input_dir={} --output_dir={} --scenario_map_dir={}'.format(
            simulation_path, source_dir, dest_dir, map_dir))
    if return_code != 0:
        logging.error('Failed to execute logsim_generator for task {}'.format(source_dir))
        # Print log here only, since rerunning will probably fail again.
        # Need people intervention instead
        # return

    # Mark complete
    complete_file = os.path.join(dest_dir, 'COMPLETE')
    logging.info('Touching complete file {}'.format(complete_file))
    if not os.path.exists(complete_file):
        os.mknod(complete_file)


class ScenarioExtractionPipeline(BasePipeline):
    """Extract logsim scenarios from records and save the bag/json pair"""

    def __init__(self):
        """Initialize"""
        BasePipeline.__init__(self, 'scenario_extraction')

    def run_test(self):
        """Local mini test."""
        root_dir = '/apollo'
        original_prefix = 'data'
        target_prefix = 'modules/data/fuel/testdata/modules/simulation/logsim_scenarios'

        # RDD(tasks), the tasks without root_dir as prefix
        todo_tasks = get_todo_tasks(
            original_prefix, target_prefix,
            lambda path: self.to_rdd(list_end_files(os.path.join(root_dir, path))))
        logging.info('todo tasks: {}'.format(todo_tasks.collect()))

        self.run(todo_tasks, original_prefix, target_prefix)
        logging.info('Simulation: All Done, TEST')

    def run_prod(self):
        """Work on actual road test data. Expect a single input directory"""
        original_prefix = 'small-records/2019'
        target_prefix = 'modules/simulation/logsim_scenarios/2019'

        # RDD(tasks)
        todo_tasks = get_todo_tasks(original_prefix, target_prefix,
                                    lambda path: self.to_rdd(self.bos().list_files(path)))
        logging.info('todo tasks: {}'.format(todo_tasks.collect()))

        self.run(todo_tasks, original_prefix, target_prefix)
        logging.info('Simulation: All Done, PROD')

    def run(self, todo_tasks, original_prefix, target_prefix):
        """Run the pipeline with given parameters"""
        # RDD(tasks)
        (todo_tasks
         # PairRDD(target_dirs, tasks), the map of target dirs and source dirs
         .keyBy(lambda source: source.replace(original_prefix, target_prefix, 1))
         # Execute each task
         .foreach(execute_task))


if __name__ == '__main__':
    ScenarioExtractionPipeline().run_test()
