import time
import os

from absl import flags

from fueling.common.base_pipeline import BasePipeline
import fueling.common.logging as logging
import yolov4.inference as inference


flags.DEFINE_string('model_path', None, 'input the model path')
flags.DEFINE_string('image_folder', None, 'input the image folder')

class AutoLabelPipeline(BasePipeline):

	def run(self):
		time_start = time.time()
		# local test use only
		# model_path = '/fuel/fueling/perception/emergency_detection/pipeline/yolov4.pth'
		# image_folder = '/fuel/fueling/perception/emergency_detection/data/FireVid/image_clips'

		# online use only
		bos_model_path = self.our_storage().abs_path(flags.FLAGS.model_path)
		logging.info(bos_model_path)
		bos_image_folder = self.our_storage().abs_path(flags.FLAGS.image_folder)
		logging.info(bos_image_folder)
		# file_list = self.our_storage().list_files(bos_image_folder, '.jpg')
		# logging.info(file_list)
		self.to_rdd(range(1)).foreach(
			lambda instance: inference.autolabel(
				bos_model_path,
				bos_image_folder
			)
		)

		#inference.autolabel(model_path, imagefolder)
		logging.info(F'Autolabel complete in {time.time() - time_start} seconds.')

if __name__ == '__main__':
    AutoLabelPipeline().main()


