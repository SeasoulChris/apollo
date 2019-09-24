#!/usr/bin/env python


from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from fueling.perception.YOLOv3.evaluate import match_label_to_result
from fueling.perception.YOLOv3.evaluate import compile_images, compile_annotations, read_results, compile_categories


def test_read_label_and_result():
    """
    Test the match_label_to_result() and read_annotation() function.
    """
    dataset_path = "/apollo/modules/data/fuel/testdata/perception/san_mateo_2018"
    result_dir = "/apollo/modules/data/fuel/testdata/perception/"
    matched_list = match_label_to_result((dataset_path, result_dir))
    a = list(map(compile_images, [matched_list]))
    b = list(map(compile_annotations, a))
    c = list(map(read_results, b))
    d = list(map(compile_categories, c)) 
    gt = d[0][0]
    dt = d[0][1]
    coco_obj = COCO()
    coco_obj.dataset = gt
    coco_obj.createIndex()
    dt_obj = coco_obj.loadRes(dt)
    estimator = COCOeval(coco_obj, dt_obj, iouType="bbox")
    estimator.evaluate()
    estimator.accumulate()
    estimator.summarize()

if __name__ == "__main__":
    test_read_label_and_result()
