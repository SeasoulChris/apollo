#!/usr/bin/env python

import os
import sys
import time

import torch

from fueling.common.base_pipeline import BasePipeline
import fueling.common.email_utils as email_utils
import fueling.common.file_utils as file_utils
from fueling.common.job_utils import JobUtils
import fueling.common.logging as logging
from fueling.learning.train_utils import cuda, train_valid_dataloader
from fueling.prediction.learning.pipelines.pedestrian_trajectory_prediction \
    .pedestrian_trajectory_dataset_cloud import PedestrianTrajectoryDatasetCloud
from fueling.prediction.learning.models.semantic_map_model.semantic_map_model \
    import SemanticMapSelfLSTMModel, WeightedSemanticMapLoss


class PedestrianTraining(BasePipeline):
    def __init__(self):
        super(PedestrianTraining, self).__init__()

    def run(self):
        self.input_path = self.FLAGS.get('input_path')
        self.input_abs_path = self.our_storage().abs_path(self.input_path)
        self.region = self.get_region_from_input_path(self.input_abs_path)
        self.data_dir = os.path.join(self.input_abs_path, 'train')
        self.model_dir_name = 'models/'
        time_start = time.time()
        self.to_rdd(range(1)).foreach(lambda instance: self.train(instance, self.data_dir))
        logging.info('Training complete in {} seconds.'.format(time.time() - time_start))

        if self.FLAGS.get('show_job_details'):
            job_id = (self.FLAGS.get('job_id') if self.is_partner_job() else
                      self.FLAGS.get('job_id')[:4])
            JobUtils(job_id).save_job_progress(100)

        self.send_email_notification(os.path.join(self.input_path, self.model_dir_name))

    def train(self, instance_id, data_dir):
        """Run training task"""
        logging.info('nvidia-smi on Executor {}:'.format(instance_id))
        if os.system('nvidia-smi') != 0:
            logging.fatal('Failed to run nvidia-smi.')
            sys.exit(-1)

        logging.info('cuda available? {}'.format(torch.cuda.is_available()))
        logging.info('cuda version: {}'.format(torch.version.cuda))
        logging.info('gpu device count: {}'.format(torch.cuda.device_count()))

        # Use gpu0 for training
        # device = torch.device('cuda:0')

        dataset = PedestrianTrajectoryDatasetCloud(data_dir, self.region)
        valid_size = dataset.__len__() // 5
        train_dataset, valid_dataset = torch.utils.data.random_split(
            dataset, [dataset.__len__() - valid_size, valid_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True,
                                                   num_workers=1, drop_last=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=4, shuffle=True,
                                                   num_workers=1, drop_last=True)
        model = SemanticMapSelfLSTMModel(30, 20).cuda()
        loss = WeightedSemanticMapLoss()
        model_file_path = file_utils.fuel_path(
            'testdata/prediction/pedestrian_semantic_lstm_torch_model.pt')
        model.load_state_dict(torch.load(model_file_path))
        learning_rate = 3e-4
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.3, patience=3, min_lr=1e-9, verbose=True, mode='min')
        # Model training:
        model_path = os.path.join(self.input_abs_path, self.model_dir_name)
        os.makedirs(model_path, exist_ok=True)
        train_valid_dataloader(train_loader, valid_loader, model, loss, optimizer, scheduler,
                               epochs=10, save_name=model_path, print_period=10, save_mode=1)
        self.jit_trace_semantic_map_model(model, model_path)

    def get_region_from_input_path(self, input_path):
        map_path = os.path.join(input_path, 'map/')
        map_list = os.listdir(map_path)
        assert len(map_list) == 1
        map_region_path = os.path.join(map_path, map_list[0])
        index = map_region_path.find('map/')
        if index == -1:
            return ''
        index += 4
        sub_path = map_region_path[index:]
        end = sub_path.find('/')
        if end == -1:
            return sub_path
        return sub_path[:end]

    def jit_trace_semantic_map_model(self, model, path):
        model.eval()
        model = model.cpu()
        X = (torch.ones([1, 3, 224, 224]), torch.ones([1, 20, 2]), torch.ones([1, 20, 2]))
        _ = model.forward(X)
        traced_cpu_model = torch.jit.trace(model.cpu(), (X,))
        traced_cpu_model.save(os.path.join(path, 'semantic_lstm_pedestrian_cpu_model.pt'))
        traced_gpu_model = torch.jit.trace(model.cuda(), (cuda(X),))
        traced_gpu_model.save(os.path.join(path, 'semantic_lstm_pedestrian_gpu_model.pt'))

    def send_email_notification(self, model_path):
        """Send email notification to users"""
        title = 'Your prediction model training job is done!'
        content = {
            'Job Owner': self.FLAGS.get('job_owner'),
            'Job ID': self.FLAGS.get('job_id'),
            'Model Path': model_path,
        }
        receivers = email_utils.PREDICTION_TEAM
        if os.environ.get('PARTNER_EMAIL'):
            receivers.append(os.environ.get('PARTNER_EMAIL'))
        email_utils.send_email_info(title, content, receivers)


if __name__ == '__main__':
    PedestrianTraining('baidusasha', '/fuel/kinglong_data/train/').main()
