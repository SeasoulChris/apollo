#!/usr/bin/env python
"""
Run with:
    bazel run //fueling/perception/emergency_detection/pipeline:record_image_parser -- --cloud
    --record_folder=<record_folder_path_on_bos>

Example:
    bazel run //fueling/perception/emergency_detection/pipeline:record_image_parser -- --cloud
    --record_folder=public-test/2020/2020-08-24/2020-08-24-14-30-55
"""
import time
import os

# Apollo packages
# from modules.drivers.proto.sensor_image_pb2 import Image
# from modules.drivers.proto.sensor_image_pb2 import CompressedImage

# Apollo-fuel packages
from fueling.common.base_pipeline import BasePipeline
import fueling.common.logging as logging
import fueling.common.record_utils as record_utils
import fueling.common.file_utils as file_utils
# import fueling.perception.decode_video as decode_video

# third-party packages
from PIL import Image
from absl import flags

flags.DEFINE_string('record_folder', None, 'input the record_folder path')
bos_original_address = "modules/perception/emergency_detection/original"
bos_processed_address = "modules/perception/emergency_detection/processed"


class RecordImageParser(BasePipeline):
    """record -> image"""

    def run(self):
        time_start = time.time()
        workers = int(os.environ.get('APOLLO_EXECUTORS', 1))
        logging.info("workers", workers)
        # local test only
        # self.parse('/fuel/fueling/perception/emergency_detection/pipeline/20200824143055.record.00006')

        # cloud only
        record_dir = flags.FLAGS.record_folder
        date_subfolder = record_dir.split("/")[3]
        file_utils.makedirs(self.our_storage().abs_path('{}/{}/images'.format(
            bos_original_address,
            date_subfolder
        )))
        file_utils.makedirs(self.our_storage().abs_path('{}/{}/compressed_images'.format(
            bos_original_address,
            date_subfolder
        )))
        file_utils.makedirs(self.our_storage().abs_path('{}/{}/images_from_video'.format(
            bos_original_address,
            date_subfolder
        )))

        self.to_rdd(
            self.our_storage().list_files(
                record_dir
            )
        ).filter(
            record_utils.is_record_file
        ).foreach(
            lambda instance: self.parse(
                instance,
                date_subfolder
            )
        )

        logging.info(F'Image pasring complete in {time.time() - time_start} seconds.')

    def parse(self, record, date_subfolder):
        record_index = record.split('.')[2]
        f12_i_count, f6_i_count, r6_i_count = 0, 0, 0
        # f12_count f6_count, r6_count,  = 0, 0, 0
        # f12_v_count, f6_v_count, r6_v_count = 0, 0, 0
        reader = record_utils.read_record([
            record_utils.FRONT_12mm_CHANNEL,
            record_utils.FRONT_12mm_VIDEO_CHANNEL,
            record_utils.FRONT_12mm_IMAGE_CHANNEL,
            record_utils.FRONT_6mm_CHANNEL,
            record_utils.FRONT_6mm_VIDEO_CHANNEL,
            record_utils.FRONT_6mm_IMAGE_CHANNEL,
            record_utils.REAR_6mm_CHANNEL,
            record_utils.REAR_6mm_VIDEO_CHANNEL,
            record_utils.REAR_6mm_IMAGE_CHANNEL
        ])

        for msg in reader(record):
            if msg.topic == record_utils.FRONT_12mm_IMAGE_CHANNEL:
                front_12mm_image = record_utils.message_to_proto(msg)
                logging.info("Processing image: front 12 mm image...")
                try:
                    im = Image.frombytes(
                        "RGB",
                        (front_12mm_image.width, front_12mm_image.height),
                        front_12mm_image.data
                    )
                    im.save(
                        self.our_storage().abs_path("{}/{}/images/{}_f12i_{}.jpg".format(
                            bos_original_address,
                            date_subfolder,
                            record_index,
                            f12_i_count
                        )),
                        "JPEG"
                    )
                except Exception as e:
                    logging.error(e)
                f12_i_count += 1

            elif msg.topic == record_utils.FRONT_6mm_IMAGE_CHANNEL:
                front_6mm_image = record_utils.message_to_proto(msg)
                logging.info("Processing image: front 6 mm image...")
                try:
                    im = Image.frombytes(
                        "RGB",
                        (front_6mm_image.width, front_6mm_image.height),
                        front_6mm_image.data
                    )
                    im.save(
                        self.our_storage().abs_path("{}/{}/images/{}_f6i_{}.jpg".format(
                            bos_original_address,
                            date_subfolder,
                            record_index,
                            f6_i_count
                        )),
                        "JPEG"
                    )
                except Exception as e:
                    logging.error(e)
                f6_i_count += 1

            elif msg.topic == record_utils.REAR_6mm_IMAGE_CHANNEL:
                rear_6mm_image = record_utils.message_to_proto(msg)
                logging.info("Processing image: rear 6 mm image...")
                try:
                    im = Image.frombytes(
                        "RGB",
                        (rear_6mm_image.width, rear_6mm_image.height),
                        rear_6mm_image.data
                    )
                    im.save(
                        self.our_storage().abs_path("{}/{}/images/{}_r6i_{}.jpg".format(
                            bos_original_address,
                            date_subfolder,
                            record_index,
                            r6_i_count
                        )),
                        "JPEG"
                    )
                except Exception as e:
                    logging.error(e)
                r6_i_count += 1


if __name__ == '__main__':
    RecordImageParser().main()
