#!/usr/bin/env python

import os

from PIL import Image, ImageDraw
from queue import Queue
from random import shuffle
from threading import Thread
import numpy as np

from fueling.perception.YOLOv3 import config as cfg
from fueling.perception.YOLOv3.utils import data_utils


BATCH_SIZE = cfg.batch_size
NUM_THREADS = cfg.num_threads


class Dataset:

    def __init__(self, image_file_paths, batch_size=BATCH_SIZE, num_threads=NUM_THREADS,
                 random_color_shift=False, random_crop_bool=False, random_jitter_bool=False,
                 output_name=False, one_shot=False):
        """
        Initialize a Dataset object. This Dataset class uses multi-threading to process input data.
        params:
        dataset_path: path to the directory that containes txt file, each txt file includes the
                      labels for 1 image. "class_name, x_min,y_min,x_max,y_max,x,x,x,x,x,x,x,x,
                      class_id, x,x,x,x,x,x,x,x"
        batch_size: number of examples per batch
        num_threads: number of thread to process data. 
        output_name: bool, if output also the image/label file names
        """
        self.image_file_paths = image_file_paths
        self.batch_size = batch_size
        self.output_name = output_name
        self.random_color_shift = random_color_shift
        self.random_crop = random_crop_bool
        self.random_jitter = random_jitter_bool
        self.one_shot = one_shot
        self.one_shot_complete = False

        self._txt_files_queue = Queue(maxsize=10000)
        self._example_queue = Queue(maxsize=100)

        self._txt_files = self.image_file_paths

        self._idx = 0
        self._num_files = len(self._txt_files)

        txt_parser_thread = Thread(target=self._parse_txt)
        txt_parser_thread.daemon = True
        txt_parser_thread.start()

        for _ in range(num_threads):
            worker = Thread(target=self._parse_example)
            worker.daemon = True
            worker.start()

    def _parse_txt(self):
        """
        Parse txt lines
        """
        while True:
            if self._idx == self._num_files:
                if self.one_shot:
                    self.one_shot_complete = True
                    break
                self._idx = 0
                shuffle(self._txt_files)
            self._txt_files_queue.put(self._txt_files[self._idx])
            self._idx += 1

    def one_shot_completed(self):
        """
        Completed one whole iteration of the dataset?"
        """
        if not self.one_shot:
            raise RuntimeError(
                "Method 'one_shot_completed' can be called only when self.one_shot is True.")
        return self.on_shot_complete

    def _parse_example(self):
        """
        Parse example from txt line.
        """
        while not self.one_shot_complete:
            image_path = self._txt_files_queue.get()
            all_paths = data_utils.get_all_paths(image_path)
            processed = data_utils.process_data(all_paths)
            # Filter out classes that is not being considered
            image_data, y_true, cls_box_map, objs, calib, original_image = \
                data_utils.filter_classes(processed)
            scale1, scale2, scale3 = y_true
            #image_data = np.expand_dims(image_data, axis=0)
            image_name = os.path.basename(image_path).split(".")[0]
            final_data = (image_data, scale1, scale2, scale3, cls_box_map, \
                          objs, calib, image_name, original_image)
            self._example_queue.put(final_data)

    @property
    def dataset_size(self):
        """
        return dataset size.
        """
        return len(self._txt_files)

    @property
    def batch(self):
        """
        Get a batch of example.
        returns:
        image_batch: np array with shape (batch_size, Input_height, Input_width, 3)
        label_batch: a list with len==batch_size
        """
        # TODO[KWT] Add support for self.one_shot
        image_batch = np.zeros(shape=(self.batch_size, cfg.Input_height, cfg.Input_width, 3),
                               dtype=np.uint8)
        label_batch_scale1 = []
        label_batch_scale2 = []
        label_batch_scale3 = []
        cls_box_map_list = []
        objs_list = []
        calib_list = []
        image_name_list = []
        original_image_list = []
        for i in range(self.batch_size):
            image_data, scale1, scale2, scale3, cls_box_map, objs, \
            calib, image_name, original_image = self._example_queue.get()
            image_batch[i] = image_data
            label_batch_scale1.append(scale1)
            label_batch_scale2.append(scale2)
            label_batch_scale3.append(scale3)
            cls_box_map_list.append(cls_box_map)
            objs_list.append(objs)
            calib_list.append(calib)
            image_name_list.append(image_name)
            original_image_list.append(original_image)

        assert not np.any(np.isnan(image_batch))
        assert not np.any(np.isnan(label_batch_scale1))
        assert not np.any(np.isnan(label_batch_scale2))
        assert not np.any(np.isnan(label_batch_scale3))

        return (image_batch,
                np.concatenate(label_batch_scale1, axis=0),
                np.concatenate(label_batch_scale2, axis=0),
                np.concatenate(label_batch_scale3, axis=0),
                cls_box_map_list,
                objs_list,
                calib_list,
                image_name_list,
                original_image_list)
