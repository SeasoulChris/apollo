"""
Run with:
    bazel run //fueling/perception/emergency_detection/pipeline:auto_label_pipeline -- --cloud
    --model_path=<model_path_in_bos>
    --image_folder=<image_folder_in_bos>

Example:
    bazel run //fueling/perception/emergency_detection/pipeline:auto_label_pipeline -- --cloud
    --model_path=modules/perception/emergency_detection/pretrained_model/yolov4.pth
    --image_folder=modules/perception/emergency_detection/data/emergency_vehicle/data/AmbulanceVid/image_clips
"""

import time
import os
import math

from absl import flags

from fueling.common.base_pipeline import BasePipeline
import fueling.common.logging as logging
import yolov4.inference as inference


flags.DEFINE_string('model_path', None, 'input the model path')
flags.DEFINE_string('image_folder', None, 'input the image folder')


class AutoLabelPipeline(BasePipeline):

    def run(self):
        time_start = time.time()
        workers = int(os.environ.get('APOLLO_EXECUTORS', 1))
        logging.info("workers", workers)

        # local test use only
        # model_path = '/fuel/fueling/perception/emergency_detection/pipeline/yolov4.pth'
        # image_folder = '/fuel/fueling/perception/emergency_detection/data/PoliceVid/image_clips'
        # file_list = self.our_storage().list_files(image_folder, ('.jpg', '.png', '.jpeg', '.bmp'))
        # print(file_list)

        # online use only
        bos_model_path = self.our_storage().abs_path(flags.FLAGS.model_path)
        logging.info("Autolabel model found at:", bos_model_path)
        bos_image_folder = self.our_storage().abs_path(flags.FLAGS.image_folder)
        logging.info("Target image folder found at:", bos_image_folder)
        img_list = self.our_storage().list_files(
            bos_image_folder, ('.jpg', '.jpeg', '.bmp', '.png'))
        logging.info("Collected images:", img_list)

        splitted_image_list = self.divide_chunks(img_list, workers)

        self.to_rdd(splitted_image_list).foreach(
            lambda instance: inference.autolabel(
                bos_model_path,
                instance
            )
        )
        logging.info(F'Autolabel complete in {time.time() - time_start} seconds.')

    def divide_chunks(self, input_list, num_of_sublist):
        ret = []
        sub_list_size = math.ceil(len(input_list) / num_of_sublist)
        for i in range(num_of_sublist):
            if i is not num_of_sublist - 1:
                ret.append(input_list[i * sub_list_size: i * sub_list_size + sub_list_size])
            else:
                ret.append(input_list[i * sub_list_size:])
        return ret


if __name__ == '__main__':
    AutoLabelPipeline().main()
