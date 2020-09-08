#!/usr/bin/env python
"""
Save audio data in records to wave files

Example command:
bazel run fueling/audio/tools:audio_saver -- --cloud --record_folder=public-test/2020/2020-08-24
"""

from absl import flags
from fueling.common.base_pipeline import BasePipeline
import fueling.common.file_utils as file_utils
import fueling.common.record_utils as record_utils
import fueling.common.logging as logging
import os
import wave

RECORD_PREFIX = 'public-test'
OUTPUT_PREFIX = 'modules/audio/wav_files_by_folder'

flags.DEFINE_string('record_folder', None, 'parse records in record folder')


class AudioSaver(BasePipeline):

    def run(self):
        # output_dir = self.our_storage().abs_path(flags.FLAGS.output_folder)
        # record_dir = self.our_storage().abs_path(flags.FLAGS.record_folder)
        # logging.info("Save to {}".format(output_dir))
        self.to_rdd(
            self.our_storage().list_files(flags.FLAGS.record_folder)).filter(
                record_utils.is_record_file).map(os.path.dirname).foreach(
                    lambda dir_path: self.save_to_wave_for_dir(dir_path))

    def save_to_wave_for_dir(self, dir_path):
        """Save records to wave files"""
        logging.info("Processing dir: {}".format(dir_path))
        self.frames = [b"" for _ in range(6)]
        reader = record_utils.read_record(
            [record_utils.MICROPHONE_CHANNEL])
        parsed_microphone = False
        records = file_utils.list_files(dir_path)
        records = sorted(records)
        for record in records:
            if not record_utils.is_record_file(record):
                continue
            for msg in reader(record):
                if msg.topic == record_utils.MICROPHONE_CHANNEL:
                    audio = record_utils.message_to_proto(msg)
                    self.parse(audio)
                    parsed_microphone = True
        if parsed_microphone:
            self.dump_to_wave_for_dir(dir_path)

    def dump_to_wave_for_dir(self, dir_path):
        """Save frame to file.wave"""
        # record_base = os.path.basename(record)
        # record_end_dir = os.path.dirname(record)
        output_end_dir = dir_path.replace(RECORD_PREFIX, OUTPUT_PREFIX)
        mkdir_cmd = 'sudo mkdir -p {}'.format(output_end_dir)
        chmod_cmd = 'sudo chmod 777 {}'.format(output_end_dir)
        os.system(mkdir_cmd)
        logging.info(mkdir_cmd)
        os.system(chmod_cmd)
        logging.info(chmod_cmd)
        file_name = os.path.basename(os.path.normpath(dir_path))
        for idx, data in enumerate(self.frames):
            if idx != 2:
                continue
            file_path = os.path.join(
                output_end_dir, "{}_channel_{}.wav".format(file_name, idx))
            with wave.open(file_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(self.sample_width)
                wf.setframerate(self.sample_rate)
                wf.writeframes(data)
            logging.info("Saved wave to {}".format(file_path))

    def save_to_wave_for_file(self, record):
        """Save records to wave files"""
        logging.info("Processing record: {}".format(record))
        self.frames = [b"" for _ in range(6)]
        reader = record_utils.read_record(
            [record_utils.MICROPHONE_CHANNEL])
        parsed_microphone = False
        for msg in reader(record):
            if msg.topic == record_utils.MICROPHONE_CHANNEL:
                audio = record_utils.message_to_proto(msg)
                self.parse(audio)
                parsed_microphone = True
        if parsed_microphone:
            self.dump_to_wave_for_file(record)

    def dump_to_wave_for_file(self, record):
        """Save frame to file.wave"""
        record_base = os.path.basename(record)
        record_end_dir = os.path.dirname(record)
        output_end_dir = record_end_dir.replace(RECORD_PREFIX, OUTPUT_PREFIX)
        mkdir_cmd = 'sudo mkdir -p {}'.format(output_end_dir)
        chmod_cmd = 'sudo chmod 777 {}'.format(output_end_dir)
        os.system(mkdir_cmd)
        logging.info(mkdir_cmd)
        os.system(chmod_cmd)
        logging.info(chmod_cmd)
        for idx, data in enumerate(self.frames):
            if idx != 2:
                continue
            file_path = os.path.join(
                output_end_dir, "{}_channel_{}.wav".format(record_base, idx))
            logging.info("Saving wave to {}".format(file_path))
            with wave.open(file_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(self.sample_width)
                wf.setframerate(self.sample_rate)
                wf.writeframes(data)

    def parse(self, audio):
        self.sample_width = audio.microphone_config.sample_width
        self.sample_rate = audio.microphone_config.sample_rate
        for idx, channel_data in enumerate(audio.channel_data):
            self.frames[idx] += channel_data.data


if __name__ == '__main__':
    AudioSaver().main()
