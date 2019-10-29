#!/usr/bin/env python3

"""Listen to the interested events and update status when those happen"""

import os

from listener import Listener
from listener import Status
import global_settings as settings
import utils as utils


class SerializeJobTaskListener(Listener):
    """SerializeJob listener that listens to file system"""

    def __init__(self):
        Listener.__init__(self)
        conf = utils.load_yaml_settings('conf/uploader_conf.yaml')
        self._src_streaming_path = conf['src_streaming_path']
        self._dst_streaming_path = conf['dst_streaming_path']
        self._for_video_only = conf['video_only']

    def get_available_task(self):
        """Check if there are any tasks that are ready to be uploaded"""
        for task in sorted(list(os.listdir(os.path.join(self._src_streaming_path, 'records')))):
            if self._task_completed(os.path.join(self._src_streaming_path, 'data'),
                                    os.path.join(self._src_streaming_path, 'records', task)):
                os.remove(os.path.join(self._src_streaming_path, 'records', task))
                self._tasks[task] = Status.INITIAL
                break
        return super().get_available_task()

    def collect(self, task_id, task, logger):
        """Collector, collects the src and dst as well as their related information"""
        collect_map = []
        task_src = utils.find_task_path(os.path.join(self._src_streaming_path, 'data'), task)
        if not task_src:
            logger.error('failed to find target {}'.format(task))
            settings.set_param(
                task_id,
                settings.get_param(task_id)._replace(ErrorCode=settings.ErrorCode.FAIL,
                                                     ErrorMsg='Target Not Found'))
            return collect_map
        if not self._task_eligible(task_src):
            settings.set_param(
                task_id,
                settings.get_param(task_id)._replace(ErrorCode=settings.ErrorCode.NOT_ELIGIBLE))
            removing_cmd = 'rm -fr {}'.format(task_src)
            logger.error('task not eligible, deleting. {}'.format(removing_cmd))
            os.system(removing_cmd)
            return collect_map
        settings.set_param(task_id, settings.get_param(task_id)._replace(Root=task_src))
        for src in os.listdir(task_src):
            src = os.path.join(task_src, src)
            dst = src.replace(self._get_root_dir(src),
                              self._get_root_dir(self._dst_streaming_path), 1)
            collect_map.append((src, len(os.listdir(src)), utils.get_size(src)), dst)
        logger.log('task id: {}, all jobs: {}'.format(task_id, collect_map))
        return collect_map

    def _get_root_dir(self, path):
        """Unify the way to find root dir"""
        return path[: path.find('modules')]

    def _task_completed(self, data_path, task_path):
        """Determine whether given task is completed"""
        root_dir = self._get_root_dir(data_path)
        records = utils.read_file_lines(task_path)
        for record_path in records:
            record_dir_in_data_path = self._record_to_stream_path(record_path, root_dir, data_path)
            if not os.path.exists(os.path.join(record_dir_in_data_path, 'COMPLETE')):
                return False
        return True

    def _task_eligible(self, task_path):
        """Check whether task is eligible"""
        check_pattern = '_apollo_sensor_camera_front_6mm_image_compressed'
        check_value = '"frame_type":"1"'
        for record in os.listdir(task_path):
            record = os.path.join(task_path, record)
            for topic_file in os.listdir(record):
                if topic_file == check_pattern:
                    file_lines = utils.read_file_lines(os.path.join(record, topic_file))
                    if not self._for_video_only:
                        return len(file_lines) > 0
                    return any(line.find(check_value) != -1 for line in file_lines)
        return False

    def _record_to_stream_path(self, record_path, root_dir, data_path):
        """Convert absolute path to the corresponding stream path"""
        if record_path.startswith(root_dir):
            relative_path = record_path.replace(root_dir + '/', '', 1).strip()
        else:
            relative_path = record_path.strip('/ \n')
        return os.path.join(data_path, relative_path)
