#!/usr/bin/env python

import os

from PIL import Image, ImageDraw
from queue import Queue
from random import shuffle
from threading import Thread
import cv2
import numpy as np

from fueling.perception.YOLOv3 import config as cfg
from fueling.perception.YOLOv3.utils import data_utils
from fueling.perception.YOLOv3.utils.yolo_utils import letterbox_image
import fueling.common.logging as logging


BATCH_SIZE = cfg.batch_size
NUM_THREADS = cfg.num_threads
INPUT_WIDTH = cfg.Input_width
INPUT_HEIGHT = cfg.Input_height


class DatasetOnlyImage:

    def __init__(self, image_file_paths, batch_size=BATCH_SIZE, num_threads=NUM_THREADS):
        """
        Initialize a Dataset object. This Dataset class uses multi-threading to process input data.
        params:
        dataset_path: path to the directory that contains txt file, each txt file includes the
                      labels for 1 image. "class_name, x_min,y_min,x_max,y_max,x,x,x,x,x,x,x,x,
                      class_id, x,x,x,x,x,x,x,x"
        batch_size: number of examples per batch
        num_threads: number of thread to process data.
        output_name: bool, if output also the image/label file names
        """
        self.image_file_paths = image_file_paths
        self.batch_size = batch_size

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
                self._idx = 0
                shuffle(self._txt_files)
            self._txt_files_queue.put(self._txt_files[self._idx])
            self._idx += 1

    def _parse_example(self):
        """
        Parse example from txt line.
        """
        while True:
            image_file_path = self._txt_files_queue.get()
            image = cv2.imread(image_file_path)
            if image is None:
                logging.warn("Failed to read image {}. Skip.".format(image_file_path))
                continue
            image = image[:, :, ::-1]
            original_image = Image.fromarray(image)
            resized_image = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))
            image_data = np.array(resized_image, dtype=np.uint8)
            image_name = os.path.basename(image_file_path).split(".")[0]
            self._example_queue.put((image_data, image_name, original_image))

    @property
    def dataset_size(self):
        """
        return dataset size.
        """
        return len(self._txt_files)

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
        image_name_list = []
        original_image_list = []
        for i in range(self.batch_size):
            image_data, image_name, original_image = self._example_queue.get()
            image_batch[i] = image_data
            image_name_list.append(image_name)
            original_image_list.append(original_image)

        assert not np.any(np.isnan(image_batch))

        return (image_batch,
                image_name_list,
                original_image_list)
