#!/usr/bin/env python

"""This script generates video files from records and decode them into images"""

import ast
import operator
import os
import time

import colored_glog as glog

from fueling.common.base_pipeline import BasePipeline
import fueling.common.s3_utils as s3_utils
import fueling.streaming.streaming_utils as streaming_utils

# The compressed channels we need to decode
VIDEO_CHANNELS = {
    'front-6mm': '/apollo/sensor/camera/front_6mm/image/compressed',
    'front-12mm': '/apollo/sensor/camera/front_12mm/image/compressed',
    'rear-6mm': '/apollo/sensor/camera/rear_6mm/image/compressed',
    'left-fisheye': '/apollo/sensor/camera/left_fisheye/image/compressed',
    'right-fisheye': '/apollo/sensor/camera/right_fisheye/image/compressed'
}

# Helper functions
def group_video_frames(message_meta):
    """Divide the video frames into groups, one group is a frame I leading multiple frames P"""
    target_topic, meta_values = message_meta
    meta_list = sorted(list(meta_values), key=operator.itemgetter(0))
    glog.info('Grouping target and topic: {} with {} messages'.format(target_topic, len(meta_list)))
    groups = list()
    frames_group = list()
    for idx, (timestamp, fields, src_path) in enumerate(meta_list):
        # TODO: pending on video compression driver
        frame_type = ast.literal_eval(fields).get('frame_type', None)
        if not frame_type:
            raise ValueError('Invalid frame type for {}'.format(target_topic))
        if frame_type == 'I' or idx == len(meta_list)-1:
            if frames_group and frames_group[0][1] == 'I':
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
    video_decoder_executable_path = '/apollo/modules/perception/decoder'
    image_output_pattern = '%05d.jpg'
    image_output_path = os.path.join(target_dir, cur_group_name)
    streaming_utils.create_dir_if_not_exist(image_output_path)
    return_code = os.system('cd {} && ./bin/video2jpg.sh --file={} --output={}'.format(
        video_decoder_executable_path, h265_video_file_path,
        os.path.join(image_output_path, image_output_pattern)))
    if return_code != 0:
        raise ValueError('Failed to execute video2jpg for video {}'.format(h265_video_file_path))
    generated_images = sorted(list(os.listdir(image_output_path)))
    if len(generated_images) != len(meta_list):
        raise ValueError('Mismatch between original frames and generated images for video {}'
                         .format(h265_video_file_path))
    # Rename the generated images to match the original frame name
    for idx in range(0, len(generated_images)):
        os.rename(os.path.join(image_output_path, generated_images[idx]),
                  os.path.join(image_output_path, streaming_utils.get_message_id(meta_list[idx][0], topic)))
    glog.info('done with group {}, image path: {}'.format(current_group, image_output_path))

def mark_complete(todo_tasks, target_dir, root_dir):
    """Create COMPLETE file to mark the job done"""
    for task in todo_tasks:
        task_path = os.path.join(root_dir, target_dir, os.path.basename(task))
        if not os.path.exists(task_path):
            glog.warn('no data generated for task: {}, \
                check if there are qualified frames in there'.format(task_path))
            continue
        streaming_utils.write_to_file(
            os.path.join(task_path, 'COMPLETE'), 'w', '{:.6f}'.format(time.time()))

class DecodeVideoPipeline(BasePipeline):
    """PopulateFrames pipeline."""

    def __init__(self):
        BasePipeline.__init__(self, 'decode-videos')

    def run_test(self):
        """Run test."""
        root_dir = '/apollo'
        target_dir = 'modules/perception/videos/decoded'
        streaming_utils.create_dir_if_not_exist(os.path.join(root_dir, target_dir))
        glog.info('Running TEST, target_dir: {}'.format(os.path.join(root_dir, target_dir)))

        _, todo_tasks = streaming_utils.get_todo_records(root_dir, target_dir)
        glog.info('ToDo tasks: {}'.format(todo_tasks))

        self.run(todo_tasks, root_dir, target_dir)

        glog.info('Task done, marking COMPLETE')
        mark_complete(todo_tasks, target_dir, root_dir)

        glog.info('Video Decoding: All Done, TEST.')

    def run_prod(self):
        """Run prod."""
        root_dir = s3_utils.BOS_MOUNT_PATH
        target_dir = 'modules/perception/videos/decoded'
        streaming_utils.create_dir_if_not_exist(os.path.join(root_dir, target_dir))
        glog.info('Running PROD, target_dir: {}'.format(os.path.join(root_dir, target_dir)))

        _, todo_tasks = streaming_utils.get_todo_records(root_dir, target_dir)
        glog.info('ToDo tasks: {}'.format(todo_tasks))

        self.run(todo_tasks, root_dir, target_dir)

        glog.info('Task done, marking COMPLETE')
        mark_complete(todo_tasks, target_dir, root_dir)

        glog.info('Video Decoding: All Done, PROD.')

    def run(self, todo_tasks, root_dir, target_dir):
        """Run the pipeline with given arguments."""
        (self.to_rdd(todo_tasks)
         # RDD(task_dir), distinct paths
         .distinct()
         # PairRDD(target_dir, task)
         .map(lambda task: (os.path.join(root_dir, target_dir, os.path.basename(task)), task))
         # PairRDD(target_dir, record)
         .flatMapValues(streaming_utils.list_records_for_task)
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
         .map(decode_videos)
         # Trigger actions
         .count())

if __name__ == '__main__':
    DecodeVideoPipeline().run_test()
