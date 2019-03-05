#!/usr/bin/env python
"""
This is a module to extraction logsim scenarios from records
based on disengage info
"""
import glob
import os

from cyber_py.record import RecordReader

from fueling.common.base_pipeline import BasePipeline
import fueling.common.colored_glog as glog
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils

from modules.routing.proto.routing_pb2 import RoutingResponse

def list_end_files(target_dir):
    """
    List all end files recursively under the specified dir.  
    This is for testing used by run_test() which simulates the production behavior
    """
    glog.info('target_dir: {}'.format(target_dir))
    end_files = list()
    for (dirpath, _, filenames) in os.walk(target_dir):
        end_files.extend([os.path.join(dirpath, file_name) for file_name in filenames])

    glog.info('end_files: {}'.format(end_files))
    return end_files

def list_completed_dirs(prefix, list_func):
    """List directories that contains COMPLETE mark up files"""
    return list_func(prefix) \
                     .filter(lambda path: path.endswith('/COMPLETE')) \
                     .map(os.path.dirname)

def get_todo_tasks(original_prefix, target_prefix, list_func):
    """Get todo tasks in rdd format."""
    original_dirs = list_completed_dirs(original_prefix, list_func)
    processed_dirs = list_completed_dirs(target_prefix, list_func) \
                        .map(lambda path: path.replace(target_prefix, original_prefix, 1))
    return original_dirs.subtract(processed_dirs)

def execute_task(task):
    """Execute task by task"""
    dest_dir, source_dir = task
    map_dir = 'modules/map/data/san_mateo'
    glog.info('Executing task with src_dir: {}, dst_dir: {}'.format(source_dir, dest_dir))

    records = list(glob.glob(source_dir + "/*.record.?????"))
    if len(records) == 0:
        glog.warning('No records file found in {}'.format(source_dir))
        return
    for msg in RecordReader(records[0]).read_messages():
        if msg.topic == '/apollo/routing_response_history':
            proto = RoutingResponse()
            proto.ParseFromString(msg.message)
            if proto.map_version.startswith('sunnyvale'):
                map_dir = 'modules/map/data/sunnyvale_with_two_offices'
            break

    # Invoke logsim_generator binary
    glog.info("Start to extract logsim scenarios for {} and map {}".format(source_dir, map_dir))
    return_code = os.system('bash {}/logsim_generator.sh {} {} {}' \
        .format(os.path.dirname(os.path.realpath(__file__)), \
                source_dir, \
                dest_dir, \
                map_dir))
    if return_code != 0:
        glog.error('Failed to execute logsim_generator for task {}'.format(source_dir))
        return
    
    # Mark complete
    complete_file = os.path.join(dest_dir, 'COMPLETE')
    glog.info('Touching complete file {}'.format(complete_file))
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

        spark_context = self.get_spark_context()
        todo_tasks = get_todo_tasks(original_prefix, \
                                    target_prefix, \
                                    lambda path: spark_context.parallelize( \
                                        list_end_files(os.path.join(root_dir, path))))
        glog.info('todo tasks: {}'.format(todo_tasks.collect()))

        self.run(todo_tasks, original_prefix, target_prefix)
        glog.info('Simulation: All Done, TEST')

    def run_prod(self, input_dir):
        """Work on actual road test data. Expect a single input directory"""
        original_prefix = 'small-records/2019'
        target_prefix = 'logsim_scenarios/2019'
        bucket = 'apollo-platform'

        todo_tasks = get_todo_tasks(original_prefix, \
                                    target_prefix, \
                                    lambda path: s3_utils.list_files(bucket, path))
        glog.info('todo tasks: {}'.format(todo_tasks.collect()))

        self.run(todo_tasks, original_prefix, target_prefix)
        glog.info('Simulation: All Done, PROD')

    def run(self, todo_tasks, original_prefix, target_prefix):
        """Run the pipeline with given parameters"""
        todo_tasks \
            .keyBy(lambda source: source.replace(original_prefix, target_prefix, 1)) \
            .map(execute_task) \
            .count()

if __name__ == '__main__':
    ScenarioExtractionPipeline().run_test()
