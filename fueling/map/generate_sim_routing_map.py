#!/usr/bin/env python
"""
This is a module to gen sim map
"""

import os
import glob

import fueling.common.logging as logging
import fueling.common.file_utils as file_utils
import fueling.common.email_utils as email_utils
from fueling.common.partners import partners
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

    routing_command = '{} --flagfile=/apollo/modules/routing/conf/routing.conf -alsologtostderr --map_dir={}'.format(
        routing_topo_creator_bin, source_dir)
    return_code = os.system(routing_command)
    logging.info("return code for gen routing map command is {}".format(return_code))

    if return_code != 0:
        logging.error('failed to generate routing map')
        return
    logging.info('Successed to generate routing map')


class SimMapPipeline(BasePipeline):

    """generate sim routing map"""

    def __init__(self):
        """Initialize"""
        BasePipeline.__init__(self, 'sim_map')

    def run_test(self):
        """Local mini test."""
        original_path = '/apollo/data'
        self.run(original_path)
        logging.info('sim map gen: Done, TEST')

    def run_prod(self):
        """Production."""
        job_owner = self.FLAGS.get('job_owner')
        job_id = self.FLAGS.get('job_id')
        logging.info("job_id: %s" % job_id)

        src_prefix = 'test/simplehdmap/data'
        origin_prefix = os.path.join(src_prefix, job_owner, job_id)
        logging.info("origin_prefix: %s" % origin_prefix)

        bos_client = BosClient()
        source_path = bos_client.abs_path(origin_prefix)
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

        receivers = email_utils.SIMPlEHDMAP_TEAM + email_utils.DATA_TEAM
        partner = partners.get(job_owner)
        if partner:
            receivers.append(partner.email)
        title = 'Your simplehdmap generated is done!'
        content = {'Job Owner': job_owner, 'Job ID': job_id}
        bin_files = glob.glob(os.path.join(origin_prefix, '*/*.txt'))
        txt_files = glob.glob(os.path.join(origin_prefix, '*/*.bin'))
        attachments = bin_files + txt_files
        logging.info('bin_files: %s' % bin_files)
        logging.info('txt_files: %s' % txt_files)
        logging.info('attachments before tar: %s' % attachments)
        # add all file to a tar.gz file
        if attachments:
            output_filename = os.path.join(origin_prefix, 'result.tar.gz')
            # with tarfile.open(output_filename, "w:gz") as tar:
            tar = tarfile.open(output_filename, 'w:gz')
            for attachment in attachments:
                hdmap = os.path.basename(os.path.dirname(attachment))
                file_name = os.path.basename(attachment)
                logging.info('add_to_tar_attachment: %s' % attachment)
                logging.info('add_to_tar_hdmap: %s' % hdmap)
                logging.info('add_to_tar_file_name: %s' % file_name)
                tar.add(attachment, arcname='%s_%s' % (hdmap, file_name))
            tar.close()
            tar = tarfile.open(output_filename, 'r:gz')
            tar.extractall(origin_prefix)
            tar.close()
            logging.info('output_filename: %s' % output_filename)
            attachments = [output_filename]
            logging.info('attachments: %s' % attachments)
        email_utils.send_email_info(title, content, receivers, attachments)

        logging.info('Simplehdmap: Done, PROD')

    def run(self, original_path):
        """Run the pipeline with given parameters"""
        # RDD(task_path)
        self.to_rdd([original_path]).foreach(execute_task)


if __name__ == '__main__':
    SimMapPipeline().main()
