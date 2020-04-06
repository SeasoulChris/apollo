#!/usr/bin/env python
import os
import re

import fueling.common.logging as logging
import fueling.common.file_utils as file_utils
from modules.planning.proto import learning_data_pb2

class DataInspector:
    '''
    A data inspector to walk through learning_data file in a folder, and show some meta informations
    '''
    def __init__(self, data_dir):

        self.instances = []
        self.total_num_instances = 0

        self._load_instances(data_dir)

        self._inspect_instances()


    def _load_instances(self, data_dir):
        logging.info('Processing directory: {}'.format(data_dir))
        all_file_paths = file_utils.list_files(data_dir)
        all_file_paths.sort(
            key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

        for file_path in all_file_paths:
            if 'future_status' not in file_path or 'bin' not in file_path:
                continue
            logging.info("loading {} ...".format(file_path))
            learning_data_frames = learning_data_pb2.LearningData()
            with open(file_path, 'rb') as file_in:
                learning_data_frames.ParseFromString(file_in.read())
            for learning_data_frame in learning_data_frames.learning_data:
                self.instances.append(learning_data_frame)
        
        self.total_num_instances = len(self.instances)

        logging.info('Total number of data points = {}'.format(
            self.total_num_instances))

    def _inspect_instances(self):
        '''
        List of inspected items includes:
            1. whether routing is given
            2. How delta_t is between labeled future trajectory and How long it is 
            2. How delta_t is between past ego trajectory and How long it is 
            2. How delta_t is an obstacle past trajectory and How long it is 
            2. How delta_t is an obstacle prediction trajectory and How long it is 
        '''
        pass

