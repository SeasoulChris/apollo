import numpy as np
from perception_pointpillars.nms import non_max_suppression

'''
try:
    from fueling.perception.pointpillars.second.core.non_max_suppression.nms import (
        non_max_suppression)
except BaseException:
    current_dir = Path(__file__).resolve().parents[0]
    load_pb11(
        ["/fuel/fueling/perception/pointpillars/second/core/cc/nms/nms_kernel.cu",
            "/fuel/fueling/perception/pointpillars/second/core/cc/nms/nms.cc"],
        current_dir / "nms.so",
        current_dir,
        cuda=True)
    from fueling.perception.pointpillars.second.core.non_max_suppression.nms import (
        non_max_suppression)
'''


def nms_gpu_cc(dets, nms_overlap_thresh, device_id=0):
    boxes_num = dets.shape[0]
    keep = np.zeros(boxes_num, dtype=np.int32)
    scores = dets[:, 4]
    order = scores.argsort()[::-1].astype(np.int32)
    sorted_dets = dets[order, :]
    num_out = non_max_suppression(sorted_dets, keep, nms_overlap_thresh,
                                  device_id)
    keep = keep[:num_out]
    return list(order[keep])


def rotate_iou_gpu_eval(boxes, query_boxes, criterion=-1, device_id=0):
    return None
