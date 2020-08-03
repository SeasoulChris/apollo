#!/usr/bin/env python
import sys
import os
sys.path.append("/fuel")

from absl import flags
from easydict import EasyDict as edict
import torch

from fueling.common.base_pipeline import BasePipeline
import fueling.common.logging as logging
from fueling.perception.emergency_detection.YOLOv4.train import train_yolov4
from fueling.perception.emergency_detection.YOLOv4.cfg import Cfg


flags.DEFINE_string('gpu_id', '0', 'GPU')
flags.DEFINE_integer('classes', 80, 'number of classes')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
flags.DEFINE_string('optimizer', 'adam', 'training optimizer')
flags.DEFINE_string('iou_type', 'iou', 'iou type (iou, giou, diou, ciou)')
#flags.DEFINE_integer('keep_checkpoint_max', 10, 'maximum number of checkpoints to keep. If set 0, all checkpoints will be kept')
flags.DEFINE_string('load', None, 'Load model from a .pth file')
flags.DEFINE_string('pretrained', None, 'pretrained yolov4.conv.137')
flags.DEFINE_string('image_dir', None, 'the directory which contains images')
flags.DEFINE_string('label_dir', '', 'the directory which contains label files')
flags.DEFINE_string('checkpoint_dir', '', 'the directory to save checkpint files')
flags.DEFINE_string('training_log_dir', '', 'the directory to save log files')


def get_args(**kwargs):
    cfg = kwargs
    args={
        'gpu': flags.FLAGS.gpu_id, 
        'classes': flags.FLAGS.classes, 
        'learning_rate': flags.FLAGS.learning_rate, 
        'load': flags.FLAGS.load, 
        'pretrained': flags.FLAGS.pretrained, 
        'dataset_dir': flags.FLAGS.image_dir, 
        'train_label': os.path.join(flags.FLAGS.label_dir, 'train.txt'), 
        'val_label': os.path.join(flags.FLAGS.label_dir, 'val.txt'), 
        'checkpoints': flags.FLAGS.checkpoint_dir,
        'TRAIN_TENSORBOARD_DIR': flags.FLAGS.training_log_dir, 
        'TRAIN_OPTIMIZER': flags.FLAGS.optimizer, 
        'iou_type': flags.FLAGS.iou_type, 
        'keep_checkpoint_max': 10
    }

    cfg.update(args)

    return edict(cfg)

class EmergencyVehicleDetector(BasePipeline):

    def run(self):
        cfg = get_args(**Cfg)
        self.to_rdd(range(1)).foreach(lambda instance: self.train(instance, cfg))

    @staticmethod
    def train(instance, cfg):

        logging.info(F'cuda available? {torch.cuda.is_available()}')
        logging.info(F'cuda version: {torch.version.cuda}')
        logging.info(F'gpu device count: {torch.cuda.device_count()}')

        logging.info(F'Config: {cfg}')

        train_yolov4(cfg, is_local=True)


if __name__ == '__main__':
    EmergencyVehicleDetector().main()
