#!/usr/bin/env python

import math

from modules.audio.proto import audio_pb2, audio_common_pb2, audio_event_pb2

from fueling.common.file_utils import list_files
import fueling.common.logging as logging
from cyber.python.cyber_py3.record import RecordReader


class AudioDetectionProfiling(object):
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.status_siren_is_on = None
        self.status_moving = None
        self.status_direction = None

        self.siren_is_on_correct_count = 0
        self.siren_is_on_wrong_count = 0
        self.moving_correct_count = 0
        self.moving_wrong_count = 0
        self.direction_correct_count = 0
        self.direction_wrong_count = 0

    def ProcessAudioEventMessage(self, audio_event):
        if audio_event.siren_is_on.Initialized():
            self.status_siren_is_on = audio_event.siren_is_on
        if audio_event.moving_result != audio_common_pb2.UNKNOWN:
            self.status_moving = audio_event.moving_result
        if audio_event.audio_direction != audio_common_pb2.UKNOWN_DIRECTION:
            self.status_direction = audio_event.audio_direction

    def DegreeToDirection(self, degree):
        if -0.25 * math.pi <= degree and degree < 0.25 * math.pi:
            return audio_common_pb2.FRONT
        if 0.25 * math.pi <= degree and degree < 0.75 * math.pi:
            return audio_common_pb2.LEFT
        if 0.75 * math.pi <= degree or degree < -0.75 * math.pi:
            return audio_common_pb2.BACK
        if -0.75 * math.pi <= degree or degree < -0.25 * math.pi:
            return audio_common_pb2.RIGHT
        return audio_common_pb2.UKNOWN_DIRECTION

    def ProcessAudioDetectionMessage(self, audio_detection):
        if self.status_siren_is_on:
            if audio_detection.is_siren:
                self.siren_is_on_correct_count += 1
            else:
                self.siren_is_on_wrong_count += 1
        if self.status_moving:
            if audio_detection.moving_result == self.status_moving:
                self.moving_correct_count += 1
            else:
                self.moving_wrong_count += 1
        if self.status_direction:
            source_degree = audio_detection.source_degree
            if self.DegreeToDirection(source_degree) == self.status_direction:
                self.direction_correct_count += 1
            else:
                self.direction_wrong_count += 1

    def Process(self):
        record_file_paths = list_files(self.dir_path)
        record_file_paths = sorted(record_file_paths)
        print(record_file_paths)

        for record_file_path in record_file_paths:
            reader = RecordReader(record_file_path)
            for message in reader.read_messages():
                if message.topic == '/apollo/audio_event':
                    audio_event = audio_event_pb2.AudioEvent()
                    audio_event.ParseFromString(message.message)
                elif message.topic == '/apollo/audio_detection':
                    audio_detection = audio_pb2.AudioDetection()
                    audio_detection.ParseFromString(message.message)
                    self.ProcessAudioDetectionMessage(audio_detection)

        logging.info('siren_is_on result (correct, wrong) = ({}, {})'.format(
                     self.siren_is_on_correct_count, self.siren_is_on_wrong_count))
        logging.info('moving result (correct, wrong) = ({}, {})'.format(
                     self.moving_correct_count, self.moving_wrong_count))


if __name__ == '__main__':
    dir_path = '/fuel/audio_bags/'
    AudioDetectionProfiling(dir_path).Process()
