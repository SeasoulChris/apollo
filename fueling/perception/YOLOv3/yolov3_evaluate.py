#!/usr/bin/env python3

from datetime import datetime
import glob
import os

from absl import flags
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import tarfile

from fueling.common.base_pipeline import BasePipeline
from fueling.common.partners import partners
from fueling.perception.YOLOv3 import config as cfg
from fueling.perception.YOLOv3.utils import yolo_utils as utils
import fueling.common.email_utils as email_utils
import fueling.common.logging as logging

CLASS_NAME_ID_MAP = cfg.class_map


def match_label_to_result(dataset_result_dir):
    """
    From labeled dataset and inference result directory,
    match each labeled txt file to its corresponding inference
    result txt file with the same name. Also append a
    unique id number to each match file pairs.
    """
    logging.info(f'currrent dataset_result_dir: {dataset_result_dir}')

    dataset_dir, result_dir = dataset_result_dir
    result_dir = os.path.join(result_dir, 'label')
    label_dir = os.path.join(dataset_dir, 'label_all')

    logging.info(f'result dir: {result_dir}, label dir: {label_dir}')

    match_list = []
    for uid, txt_name in enumerate(os.listdir(label_dir)):
        label_path = os.path.join(label_dir, txt_name)
        result_path = os.path.join(result_dir, txt_name)
        if not os.path.exists(result_path):
            continue
        match_list.append((label_path, result_path, uid))
    return match_list


def compile_images(label_result_uid_list):
    """
    Compile the images of annotations into the cocoAPI format.
    """
    image_label_result_uid_list = []
    for label_result_uid in label_result_uid_list:
        label_txt_path, result_txt_path, uid = label_result_uid
        label_dir_path, txt_name = os.path.split(label_txt_path)
        image_dir_path = os.path.join(os.path.split(label_dir_path)[0], "images_all")
        txt_name = txt_name.split(".")[0]
        if os.path.exists(os.path.join(image_dir_path, txt_name + ".jpg")):
            image_name = txt_name + ".jpg"
            width, height = Image.open(os.path.join(image_dir_path, image_name)).size
        elif os.path.exists(os.path.join(image_dir_path, txt_name + ".png")):
            image_name = txt_name + ".png"
            width, height = Image.open(os.path.join(image_dir_path, image_name)).size
        else:
            raise Exception(f'Image {os.path.join(image_dir_path, txt_name)} does not exist.')

        image_info = {}
        image_info["coco_url"] = None
        image_info["date_captured"] = None
        image_info["flickr_url"] = None
        image_info["license"] = None
        image_info["file_name'"] = image_name
        image_info["height"] = height
        image_info["width"] = width
        image_info["id"] = uid
        image_label_result_uid_list.append(
            (image_info, label_txt_path, result_txt_path, uid))
    return image_label_result_uid_list


def compile_annotations(image_label_result_uid_list):
    """
    Compile annotations into the cocoAPI format.
    """
    image_objs_label_result_uid_list = []
    obj_id = 0
    for element in image_label_result_uid_list:
        image_info, label_txt_path, result_txt_path, uid = element
        objs = []
        for line in open(label_txt_path).readlines():
            cls_name = line.split()[0]
            if cls_name in CLASS_NAME_ID_MAP:
                objs += [([float(e) for e in line.split()[1:]], CLASS_NAME_ID_MAP[line.split()[0]])]
        width = image_info["width"]
        height = image_info["height"]
        ann_list = []
        for obj in objs:
            x0, y0, x1, y1 = obj[0][3:7]
            # TODO[KWT]: uncomment below to clip out-of-image-bound bboxes
            # x0 = round(max(0, min(width, x0)), 3)
            # x1 = round(max(0, min(width, x1)), 3)
            # y0 = round(max(0, min(height, y0)), 3)
            # y1 = round(max(0, min(height, y1)), 3)
            cat_id = obj[1]

            obj_dic = {"area": round((x1 - x0 + 1) * (y1 - y0 + 1), 2),
                       "bbox": [x0, y0, round(x1 - x0 + 1, 2), round(y1 - y0 + 1, 2)],
                       "category_id": cat_id,
                       "id": obj_id,
                       "image_id": uid,
                       "iscrowd": 0,
                       "segmentation": None}
            obj_id += 1
            ann_list.append(obj_dic)
        image_objs_label_result_uid_list.append(
            (image_info, ann_list, label_txt_path, result_txt_path, uid))
    return image_objs_label_result_uid_list


def read_results(image_objs_label_result_uid_list):
    """
    Read from label and result txt files, to numpy matrix of boxes.
    Input label_result_uid should be the element of the output of function
    match_label_to_result().
    """
    def _read_into_matrix(txt_file):
        return_matrix = []
        with open(txt_file, "r") as handle:
            lines = handle.readlines()
            for line in lines:
                line = line.split(" ")
                if line[0] in CLASS_NAME_ID_MAP:
                    object_instance = [0.0 for _ in range(7)]
                    object_instance[0] = uid
                    object_instance[1] = float(line[4])
                    object_instance[2] = float(line[5])
                    object_instance[3] = float(line[6]) - float(line[4])
                    object_instance[4] = float(line[7]) - float(line[5])
                    object_instance[5] = float(line[15])
                    object_instance[6] = CLASS_NAME_ID_MAP[line[0]]
                    return_matrix.append(np.asarray(object_instance))
        if return_matrix:
            np.vstack(return_matrix)
            return np.asarray(return_matrix)
        return None

    image_info_list = []
    ann_all_list = []
    result_matrix = []
    for element in image_objs_label_result_uid_list:
        image_info, ann_list, label_txt_path, result_txt_path, uid = element
        result_matrix_np = _read_into_matrix(result_txt_path)
        if result_matrix_np is not None:
            result_matrix.append(result_matrix_np)
            image_info_list.append(image_info)
            ann_all_list += ann_list
    gt_dict = {"annotations": ann_all_list,
               "images": image_info_list}
    complete_result_matrix = np.vstack(result_matrix) if result_matrix else []

    return (gt_dict, complete_result_matrix)


def compile_categories(gt_dt):
    """
    Compile categories into the cocoAPI format.
    """
    gt, dt = gt_dt
    cat_list = []
    for k, v in CLASS_NAME_ID_MAP.items():
        cat_list.append({"id": v,
                         "name": k,
                         "supercategory": k})
    gt["categories"] = cat_list
    return (gt, dt)


def notify_results(result_dir, stats, job_owner, job_id):
    """Send email to partners for notification"""

    result_file_path = os.path.join(result_dir,
                                    f'{datetime.today().strftime("%Y-%m-%d-%H-%M-%S")}-metrics.txt')
    with open(result_file_path, 'w') as result_file:
        result_file.write('\n'.join(stats))

    title = 'Your model training job is done!'
    receivers = email_utils.DATA_TEAM + email_utils.PERCEPTION_TEAM
    partner = partners.get(job_owner)
    if partner:
        receivers.append(partner.email)
    content = {'Job Owner': job_owner, 'Job ID': job_id}

    tar_file_path = f'{result_file_path}.tar.gz'
    tar = tarfile.open(tar_file_path, 'w:gz')
    tar.add(result_file_path, os.path.basename(result_file_path))
    tar.close()

    email_utils.send_email_info(title, content, receivers, [tar_file_path])


class Yolov3Evaluate(BasePipeline):
    """Evaluate pipeline."""

    def run_test(self):
        """Run test."""
        datasets_dir = glob.glob(
            os.path.join('/apollo/modules/data/fuel/testdata/perception/YOLOv3/train', '*'))
        datasets = [(dataset, '/apollo/modules/data/fuel/testdata/perception/YOLOv3/test_output/san_mateo_2018')
                    for dataset in datasets_dir]
        self.run(datasets)

    def run_prod(self):
        """Run prod."""
        object_storage = self.partner_storage() or self.our_storage()
        ground_truth_path = object_storage.abs_path(self.FLAGS.get('input_data_path'))
        inference_data_path = object_storage.abs_path(self.FLAGS.get('output_data_path'))
        inference_data_path = os.path.join(inference_data_path, cfg.inference_path)
        self.run([(ground_truth_path, inference_data_path)])

    def run(self, datasets):
        """Run the actual pipeline job."""

        def _executor(gt_dt):
            """Executor task that runs on workers"""
            gt, dt = gt_dt
            coco_obj = COCO()
            coco_obj.dataset = gt
            coco_obj.createIndex()
            dt_obj = coco_obj.loadRes(dt)
            evaluator = COCOeval(coco_obj, dt_obj, iouType="bbox")
            evaluator.evaluate()
            evaluator.accumulate()
            stats = utils.summarize(evaluator)
            notify_results(datasets[0][1], stats,
                           self.FLAGS.get('job_owner'), self.FLAGS.get('job_id'))

        data_rdd = (
            # RDD((label_dataset, result_dir)), each dataset to be evaluatued
            self.to_rdd(datasets)
            # RDD([(label_txt_path, result_txt_path, uid),...]), list all txt files
            # in a dataset and the result directory
            .map(match_label_to_result)
            # RDD([(image_dic, label_txt_path, result_txt_path, uid),...]), add the
            # image information as a dictionary
            .map(compile_images)
            # RDD([(image_dict, ann_list, label_txt_path, result_txt_path, uid),...]),
            # add the annotation list for that image example
            .map(compile_annotations)
            # RDD((gt_dict, complete_result_matrix)), consolidate the list into a
            # ground truth dictionary and an inference result matrix
            .map(read_results)
            # RDD((gt_dict, complete_result_matrix)), add the category list to the
            # gt_dict
            .map(compile_categories)
        )

        data_rdd.foreach(_executor)


if __name__ == "__main__":
    Yolov3Evaluate().main()
