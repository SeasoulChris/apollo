#!/usr/bin/env python
"""
Save audio data in records to wave files

Run with:
    bazel run //fueling/demo:stat_auto_mileage -- --cloud
    --record_folder=<records_dir_path_in_bos>
    --output_folder=<output_path_for_wave_in_bos>
"""

from absl import flags
from fueling.common.base_pipeline import BasePipeline
import fueling.common.record_utils as record_utils
import fueling.common.logging as logging
import os
import wave

# RECORD_PREFIX = 'public-test/2020/2020-08-24/2020-08-24-14-58-13'
# OUTPUT_DIR = 'modules/audio/waves'

flags.DEFINE_string('record_folder', None, 'parse records in record folder')
flags.DEFINE_string('output_folder', None, 'save wave files to output folder')


class AudioSaver(BasePipeline):

    def run(self):
        output_dir = self.our_storage().abs_path(flags.FLAGS.output_folder)
        logging.info("Save to {}".format(output_dir))
        self.to_rdd(
            self.our_storage().list_files(flags.FLAGS.record_folder)).filter(
                record_utils.is_record_file).foreach(
                    lambda r: self.save_to_wave(r, output_dir))

    def save_to_wave(self, record, output_dir):
        """Save records to wave files"""
        logging.info("Processing record: {}".format(record))
        self.frames = [b"" for _ in range(6)]
        reader = record_utils.read_record(
            [record_utils.MICROPHONE_CHANNEL])
        for msg in reader(record):
            if msg.topic == record_utils.MICROPHONE_CHANNEL:
                audio = record_utils.message_to_proto(msg)
                self.parse(audio)
        self.dump_to_wave(os.path.basename(record), output_dir)

    def dump_to_wave(self, record_base, output_dir):
        """Save frame to file.wave"""
        for idx, data in enumerate(self.frames):
            if idx != 2:
                continue
            file_path = os.path.join(
                output_dir, "{}_channel_{}.wav".format(record_base, idx))
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
