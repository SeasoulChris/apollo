#!/usr/bin/env python
"""
This is a module to gen sim map
"""

import os
import glob

import fueling.common.logging as logging
from fueling.common.base_pipeline import BasePipeline
from fueling.common.storage.bos_client import BosClient


def execute_task(source_dir):
    """Execute task by task"""
    logging.info('executing task with src_dir: {}'.format(source_dir))

    sim_map_generator_bin = '/apollo/bazel-bin/modules/map/tools/sim_map_generator'
    sim_command = '{} --map_dir={} --output_dir={}'.format(
        sim_map_generator_bin, source_dir, source_dir)
    logging.info('sim_map_generator command is {}'.format(sim_command))
    return_code = os.system(sim_command)
    logging.info("return code for sim_map_gen is {}".format(return_code))

    if return_code != 0:
        logging.error('failed to generate sim map')
        return
    logging.info('Successed to generate sim map')

    routing_topo_creator_bin = '/apollo/bazel-bin/modules/routing/topo_creator/topo_creator'

    routing_command = ('{} --flagfile=/apollo/modules/routing/conf/routing.conf '
                       '-alsologtostderr --map_dir={}'.format(routing_topo_creator_bin, source_dir))
    return_code = os.system(routing_command)
    logging.info("return code for gen routing map command is {}".format(return_code))

    if return_code != 0:
        logging.error('failed to generate routing map')
        return
    logging.info('Successed to generate routing map')


class SimMapPipeline(BasePipeline):
    """generate sim routing map"""

    def run_test(self):
        """Local mini test."""
        original_path = '/apollo/data'
        self.run(original_path)
        logging.info('sim map gen: Done, TEST')

    def run_prod(self):
        """Production."""
        dir_prefix = self.FLAGS.get('input_data_path', 'test/simplehdmap')
        job_owner = self.FLAGS.get('job_owner')
        job_id = self.FLAGS.get('job_id')
        logging.info("job_id: %s" % job_id)

        #src_prefix = 'simplehdmap/result'
        src_prefix = os.path.join(dir_prefix, 'result')
        origin_prefix = os.path.join(src_prefix, job_owner, job_id)
        logging.info("origin_prefix: %s" % origin_prefix)

        bos_client = BosClient()
        object_storage = self.partner_object_storage() or bos_client
        source_path = object_storage.abs_path(origin_prefix)
        logging.info('source_path path is {}'.format(source_path))
        base_map_path = os.path.join(source_path, 'base_map.txt')

        if not os.path.exists(base_map_path):
            logging.warning('base_map.txt: {} not exists'.format(base_map_path))
        sim_map_generator_path = '/apollo/bazel-bin/modules/map/tools/sim_map_generator'
        if not os.path.exists(sim_map_generator_path):
            logging.warning('sim_map_generator: {} not exists'.format(sim_map_generator_path))
        routing_creator_path = '/apollo/bazel-bin/modules/routing/topo_creator/topo_creator'
        if not os.path.exists(routing_creator_path):
            logging.warning('topo_creator: {} not exists'.format(routing_creator_path))
        # RDD(tasks), the tasks without src_prefix as prefix

        self.run(source_path)

    def run(self, original_path):
        """Run the pipeline with given parameters"""
        # RDD(task_path)
        self.to_rdd([original_path]).foreach(execute_task)


if __name__ == '__main__':
    SimMapPipeline().main()
