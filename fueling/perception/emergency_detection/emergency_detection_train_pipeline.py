#!/usr/bin/env python
import sys
sys.path.append("/fuel")

from absl import flags
from easydict import EasyDict as edict

from fueling.common.base_pipeline import BasePipeline
import fueling.common.logging as logging

from fueling.perception.emergency_detection.YOLOv4.train import train_yolov4
from fueling.perception.emergency_detection.YOLOv4.cfg import Cfg


from absl import flags
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
flags.DEFINE_string('load', None, 'Load model from a .pth file')
flags.DEFINE_integer('gpu_id', -1, 'GPU')
flags.DEFINE_string('data_dir', None, 'dataset dir')
flags.DEFINE_string('pretrained', None, 'pretrained yolov4.conv.137')
flags.DEFINE_integer('classes', 80, 'number of classes')
flags.DEFINE_string('train_label_path', 'train.txt', 'train label path')
flags.DEFINE_string('optimizer', 'adam', 'training optimizer')
flags.DEFINE_string('iou_type', 'iou', 'iou type (iou, giou, diou, ciou)')
flags.DEFINE_integer('keep_checkpoint_max', 10, 'maximum number of checkpoints to keep. If set 0, all checkpoints will be kept')

def get_args(**kwargs):
    cfg = kwargs
    
    '''
    parser = argparse.ArgumentParser(description='Train the Model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,
    #                     help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='learning_rate')
    parser.add_argument('-f', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('-g', '--gpu', metavar='G', type=str, default='-1',
                        help='GPU', dest='gpu')
    parser.add_argument('-dir', '--data-dir', type=str, default=None,
                        help='dataset dir', dest='dataset_dir')
    parser.add_argument('-pretrained', type=str, default=None, help='pretrained yolov4.conv.137')
    parser.add_argument('-classes', type=int, default=80, help='dataset classes')
    parser.add_argument('-train_label_path', dest='train_label', type=str, default='train.txt', help="train label path")
    parser.add_argument(
        '-optimizer', type=str, default='adam',
        help='training optimizer',
        dest='TRAIN_OPTIMIZER')
    parser.add_argument(
        '-iou-type', type=str, default='iou',
        help='iou type (iou, giou, diou, ciou)',
        dest='iou_type')
    parser.add_argument(
        '-keep-checkpoint-max', type=int, default=10,
        help='maximum number of checkpoints to keep. If set 0, all checkpoints will be kept',
        dest='keep_checkpoint_max')
    args = vars(parser.parse_args())
    '''

    args={'learning_rate': 0.001, 'load': None, 'gpu': '0', 'dataset_dir': '/mnt/bos/modules/perception/emergency_detection/data/emergency_vehicle/images', 
    'pretrained': '/mnt/bos/modules/perception/emergency_detection/pretrained_model/yolov4.conv.137.pth', 'classes': 2, 
    'train_label': '/mnt/bos/modules/perception/emergency_detection/data/emergency_vehicle/train.txt', 
    'val_label': '/mnt/bos/modules/perception/emergency_detection/data/emergency_vehicle/val.txt', 
    'checkpoints': '/mnt/bos/modules/perception/emergency_detection/checkpoints', 
    'TRAIN_TENSORBOARD_DIR': '/mnt/bos/modules/perception/emergency_detection/log', 
    'TRAIN_OPTIMIZER': 'adam', 'iou_type': 'iou', 'keep_checkpoint_max': 10}
    


    # for k in args.keys():
    #     cfg[k] = args.get(k)
    cfg.update(args)

    return edict(cfg)

class EmergencyVehicleDetector(BasePipeline):
    """Demo pipeline."""

    def run(self):
        #train_yolov4()
        self.to_rdd(range(1)).foreach(self.train)

    @staticmethod
    def train(instance_id):
        cfg = get_args(**Cfg)

        logging.info(F'cuda available? {torch.cuda.is_available()}')
        logging.info(F'cuda version: {torch.version.cuda}')
        logging.info(F'gpu device count: {torch.cuda.device_count()}')
        logging.info(F'instance: {instance}, world_size: {world_size}, job_id: {job_id}')

        logging.info(F'Config: {cfg}')

        train_yolov4(cfg)


if __name__ == '__main__':
    EmergencyVehicleDetector().main()
    #train_yolov4()
