#!/usr/bin/env python

"""This script generates video files from records and decode them into images"""

import ast
import glob
import operator
import os
import time

import colored_glog as glog
import cv2
import pyspark_utils.helper as spark_helper

from cyber_py.record import RecordReader, RecordWriter
from modules.drivers.proto.sensor_image_pb2 import CompressedImage

from fueling.common.base_pipeline import BasePipeline
import fueling.common.bos_client as bos_client
import fueling.common.file_utils as file_utils
import fueling.streaming.streaming_utils as streaming_utils

# The compressed channels that have videos we need to decode
VIDEO_CHANNELS = {
    'front-6mm': '/apollo/sensor/camera/front_6mm/image/compressed',
    'front-12mm': '/apollo/sensor/camera/front_12mm/image/compressed',
    'rear-6mm': '/apollo/sensor/camera/rear_6mm/image/compressed',
    'left-fisheye': '/apollo/sensor/camera/left_fisheye/image/compressed',
    'right-fisheye': '/apollo/sensor/camera/right_fisheye/image/compressed'
}

class DecodeVideoPipeline(BasePipeline):
    """PopulateFrames pipeline."""

    def __init__(self):
        BasePipeline.__init__(self, 'decode-videos')

    def run_test(self):
        """Run test."""
        root_dir = '/apollo'
        video_dir = 'modules/perception/videos/decoded'
        decoded_records_dir = 'decoded_records'

        _, todo_tasks = streaming_utils.get_todo_records(root_dir, decoded_records_dir)
        glog.info('ToDo tasks: {}'.format(todo_tasks))

        self.run(todo_tasks, root_dir, video_dir, decoded_records_dir)

        glog.info('Task done, marking COMPLETE')
        mark_video_complete(todo_tasks, video_dir, root_dir)
        mark_decoded_dir_complete(todo_tasks, decoded_records_dir, root_dir)

        glog.info('Video Decoding: All Done, TEST.')

    def run_prod(self):
        """Run prod."""
        root_dir = bos_client.BOS_MOUNT_PATH
        video_dir = 'modules/perception/videos/decoded'
        decoded_records_dir = 'decoded_records'

        _, todo_tasks = streaming_utils.get_todo_records(root_dir, decoded_records_dir)
        glog.info('ToDo tasks: {}'.format(todo_tasks))

        self.run(todo_tasks, root_dir, video_dir, decoded_records_dir)

        glog.info('Task done, marking COMPLETE')
        mark_video_complete(todo_tasks, video_dir, root_dir)
        mark_decoded_dir_complete(todo_tasks, decoded_records_dir, root_dir)

        glog.info('Video Decoding: All Done, PROD.')

    def run(self, todo_tasks, root_dir, target_dir, decoded_records_dir):
        """Run the pipeline with given arguments."""
        target_records = spark_helper.cache_and_log(
            'Task-Records',
            self.to_rdd(todo_tasks)
            # RDD(task_dir), distinct paths
            .distinct()
            # PairRDD(target_dir, task)
            .map(lambda task: (os.path.join(root_dir, target_dir, os.path.basename(task)), task))
            # PairRDD(target_dir, record)
            .flatMapValues(streaming_utils.list_records_for_task))

        # Retrieve video frames from original records, and decode them to images
        # PairRDD(target_dir, record)
        (target_records
         # PairRDD(target_dir, MessageMetaData(topic, timestamp, fields, src_path))
         .flatMapValues(lambda record: streaming_utils.load_meta_data(
             root_dir, record, VIDEO_CHANNELS.values()))
         # PairRDD((target_dir, topic), (timestamp, fields, src_path))
         .map(lambda (target, (topic, time, fields, src_path)):
              ((target, topic), (time, fields, src_path)))
         # PairRDD((target_dir, topic), (timestamp, fields, src_path)s)
         .groupByKey()
         # PairRDD((target_dir, topic), (timestamp, fields, src_path)s), cut into smaller groups
         .flatMap(group_video_frames)
         # PairRDD((target_dir, topic), (timestamp, fields, src_path)s), actually decoding
         .foreach(decode_videos))

        # Replace video frames with the decoded images back to original records
        # PairRDD(target_dir, record)
        (target_records
         .foreach(lambda target_record:
                  replace_images(target_record, root_dir, decoded_records_dir)))

# Helper functions
def group_video_frames(message_meta):
    """Divide the video frames into groups, one group is a frame I leading multiple frames P"""
    target_topic, meta_values = message_meta
    target, _ = target_topic
    if os.path.exists(os.path.join(target, 'COMPLETE')):
        glog.info('target already processed: {}, do nothing'.format(target_topic))
        return [(target_topic, [])]
    meta_list = sorted(list(meta_values), key=operator.itemgetter(0))
    glog.info('Grouping target topic: {}, with {} messages'.format(target_topic, len(meta_list)))
    groups = list()
    frames_group = list()
    for idx, (timestamp, fields, src_path) in enumerate(meta_list):
        frame_type = ast.literal_eval(fields).get('frame_type', None)
        if not frame_type:
            raise ValueError('Invalid frame type for {}'.format(target_topic))
        if frame_type == '1' or idx == len(meta_list) - 1:
            if frames_group and frames_group[0][1] == '1':
                glog.info('generating new group, size:{}'.format(len(frames_group)))
                groups.append(frames_group)
            frames_group = list()
        frames_group.append((timestamp, frame_type, src_path))
    glog.info('total groups count:{}'.format(len(groups)))
    return [(target_topic, group) for group in groups]

def decode_videos(message_meta):
    """
    Merge the video frames together to form one video file.
    Then call execuable to convert them into images
    """
    target_topic, meta_values = message_meta
    meta_list = sorted(list(meta_values), key=operator.itemgetter(0))
    glog.info('decoding task {} with {} frames:{}'.format(target_topic, len(meta_list), meta_list))
    if not meta_list:
        glog.error('no video frames for target dir and topic {}'.format(target_topic))
        return
    target_dir, topic = target_topic
    streaming_utils.create_dir_if_not_exist(target_dir)
    # Use the first message name in the group as the current group name
    cur_group_name = streaming_utils.get_message_id(meta_list[0][0], topic)
    h265_video_file_path = os.path.join(target_dir, '{}.h265'.format(cur_group_name))
    glog.info('current video file path: {}'.format(h265_video_file_path))
    with open(h265_video_file_path, 'wb') as h265_video_file:
        for _, _, video_frame_bin_path in meta_list:
            with open(video_frame_bin_path, 'rb') as video_frame_bin:
                h265_video_file.write(video_frame_bin.read())
    # Invoke video2jpg binary executable
    image_output_path = os.path.join(target_dir, cur_group_name)
    streaming_utils.create_dir_if_not_exist(image_output_path)
    video_decoder_path = '/apollo/bazel-bin/modules/drivers/video/tools/decode_video/video2jpg'
    return_code = os.system('{} --input_video={} --output_dir={}'.format(
        video_decoder_path, h265_video_file_path, image_output_path))
    if return_code != 0:
        raise ValueError('Failed to execute video2jpg for video {}'.format(h265_video_file_path))
    generated_images = sorted(glob.glob('{}/*.jpg'.format(image_output_path)))
    if len(generated_images) != len(meta_list):
        raise ValueError('Mismatch between original frames:{} and generated images:{} for video {}'
                         .format(len(meta_list), len(generated_images), h265_video_file_path))
    # Rename the generated images to match the original frame name
    for idx in range(0, len(generated_images)):
        os.rename(os.path.join(image_output_path, generated_images[idx]),
                  os.path.join(image_output_path,
                               streaming_utils.get_message_id(meta_list[idx][0], topic)))
    glog.info('done with group {}, image path: {}'.format(cur_group_name, image_output_path))

def replace_images(target_record, root_dir, decoded_records_dir):
    """Scan messages in original record file, and replace video frames with decoded image frames"""
    video_dir, record = target_record
    dst_record = locate_target_decoded_record(root_dir, decoded_records_dir, record)
    glog.info("replacing frames for {} to {}".format(target_record, dst_record))
    reader = RecordReader(record)
    file_utils.makedirs(os.path.dirname(dst_record))
    writer = RecordWriter(0, 0)
    writer.open(dst_record)
    topic_descs = {}
    counter = 0
    for message in reader.read_messages():
        if message.topic in VIDEO_CHANNELS.values():
            message_content = get_image_back(video_dir, message)
            if not message_content:
                # For any reason it failed to convert, just ignore the message
                glog.error('failed to convert message {}-{} in record {}'.format(
                    message.topic, message.timestamp, record))
                continue
        counter += 1
        if counter % 100 == 0:
            glog.info('writing {} th message to record {}'.format(counter, dst_record))
        writer.write_message(message.topic, message_content, message.timestamp)
        if message.topic not in topic_descs:
            topic_descs[message.topic] = reader.get_protodesc(message.topic)
            writer.write_channel(message.topic, message.data_type, topic_descs[message.topic])
    writer.close()

def get_image_back(video_dir, message):
    """Actually change the content of message from video bytes to image bytes"""
    message_id = streaming_utils.get_message_id(message.timestamp, message.topic)
    message_path = find_message_in_video_dir(video_dir, message_id)
    if not message_path:
        glog.error('message {} not found in video dir {}'.format(message_id, video_dir))
        return None
    img_bin = cv2.imread(message_path)
    # Check by using NoneType explicitly to avoid ambitiousness
    if img_bin is None:
        glog.error('failed to read original message: {}'.format(message_path))
        return None
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    result, encode_img = cv2.imencode('.jpg', img_bin, encode_param)
    if not result:
        glog.error('failed to encode message {}'.format(message_id))
        return None
    message_proto = CompressedImage()
    message_proto.ParseFromString(message.message)
    message_proto.format = '; jpeg compressed bgr8'
    message_proto.data = message_proto.data.replace(message_proto.data[:], bytearray(encode_img))
    return message_proto.SerializeToString()

def find_message_in_video_dir(target_dir, message_id):
    """Walk through video dir to find the corresponding decoded message"""
    for (root, _, files) in os.walk(target_dir):
        target_file = next((end_file for end_file in files if end_file == message_id), None)
        if target_file:
            return os.path.join(root, target_file)
    return None

def locate_target_decoded_record(root_dir, decoded_records_dir, record):
    """Determine the target dir of decoded records"""
    if record.find(root_dir) < 0:
        # It's possible that original record path does not start with root_dir path,
        # in this case just append the original record to decoded records dir
        dst_record = os.path.join(root_dir, decoded_records_dir, record[1:])
    else:
        # Otherwise, replace the exact next sub dir of root_dir and with decoded_records_dir
        dst_record = '{}{}'.format(os.path.join(root_dir, decoded_records_dir),
                                   record[record.find('/', len(root_dir) + 1) : ])
    return dst_record

def mark_video_complete(todo_tasks, video_dir, root_dir):
    """Create COMPLETE file to mark the video decoding part done"""
    for task in todo_tasks:
        task_path = os.path.join(root_dir, video_dir, os.path.basename(task))
        if not os.path.exists(task_path):
            glog.warn('no video decoded for task: {}'.format(task_path))
            continue
        mark_complete(task_path)

def mark_decoded_dir_complete(todo_tasks, decoded_dir, root_dir):
    """Create COMPLETE file to mark the decoded dir part done"""
    for task in todo_tasks:
        record = next((record for record in streaming_utils.list_records_for_task(task)), None)
        if not record:
            glog.warn('no record found for task: {}'.format(task))
            continue
        task_path = os.path.dirname(locate_target_decoded_record(root_dir, decoded_dir, record))
        if not os.path.exists(task_path):
            glog.warn('no decoded dir for task: {}'.format(task_path))
            continue
        mark_complete(task_path)

def mark_complete(task_path):
    """Create COMPLETE file to mark the job done"""
    streaming_utils.write_to_file(
        os.path.join(task_path, 'COMPLETE'), 'w', '{:.6f}'.format(time.time()))

if __name__ == '__main__':
    DecodeVideoPipeline().run_test()
