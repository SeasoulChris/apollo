#!/usr/bin/env python

import numpy as np


# =============== NETWORK SETTINGS ================
Input_width = 1440  # must be multiple of 32
Input_height = 800  # must be multiple of 32
original_width = 1920
original_height = 1080
visual_scale = 2  # scale to up/down sample image for training visualization
PR_curve_width = 1280
PR_curve_height = 960
channels = 3  # RBG
num_classes = 8
num_angle_bins = 1
bin_overlap_fraction = 1 / 6   # The final overlap will be 2*bin_overlap_fraction
num_output_layers = 5 + 4 * num_classes + 3 * num_angle_bins

# =================== DATA FILTER SETTINGS =====================
truncation_rate = 10    # objs with truncation rate higher than this will be filtered out
occlusion = 5           # objs with occlusion code higher than this will be filtered out

anchors = np.array([[284, 181], [426, 340], [734, 545],  # [116,90], [156, 198], [373, 326],
                    [94, 193], [177, 102], [153, 250],  # [30, 61], [62, 45], [59, 119],
                    [42, 79], [68, 38], [18, 18]])  # [10, 13], [16, 30], [33, 23]])
anchor_mask = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
num_anchor_boxes_per_scale = len(anchor_mask[0])

# ========= NMS SETTINGS ============
nms_confidence_threshold = 0.9
nms_iou_threshold = 0.3

# ===================== TRAINING AND INFERENCE SETTINGS ======================
inference = True  # Set to False for training and True for inference
inference_only_2d = False

restore_training = False
start_from_coco = False
#restore_path = "testdata/perception/YOLOv3/models/models-119999"
restore_path = "/apollo/modules/data/fuel/testdata/perception/YOLOv3/models/models-51000"
#model_output_path = "/media/2tb/kawai_data/Beijing_labeled_2019_Q1/front_6mm_2d3d_Q1_2019/san_mateo_2018/models"
#model_output_path = "/apollo/modules/data/fuel/testdata/perception/YOLOv3/models"
inference_output_path = "./testdata/perception/YOLOv3/infer_output/models-51000/"

#train_data_dir_local = "/media/2tb/kawai_data/Beijing_labeled_2019_Q1/front_6mm_2d3d_Q1_2019"
train_data_dir_local = "/apollo/modules/data/fuel/testdata/perception/YOLOv3/train"
inference_data_dir_local = "/apollo/modules/data/fuel/testdata/perception/YOLOv3/train"
evaluate_data_dir_local = "/apollo/modules/data/fuel/testdata/perception/YOLOv3/train"
evaluate_result_dir_local = "/apollo/modules/data/fuel/testdata/perception/YOLOv3/test_output"
#train_data_dir_cloud = "modules/perception/camera_object"
#inference_data_dir_cloud = "modules/perception/camera_object"
#evaluate_data_dir_cloud = "modules/perception/camera_object"
model_output_path = "/mnt/bos/modules/perception/front_6mm_2d3d_Q1_2019/"
train_data_dir_cloud = "modules/perception/front_6mm_2d3d_Q1_2019"
inference_data_dir_cloud = "modules/perception/front_6mm_2d3d_Q1_2019"
evaluate_data_dir_cloud = "modules/perception/front_6mm_2d3d_Q1_2019"
evaluate_result_dir_cloud = "modules/perception/front_6mm_2d3d_Q1_2019/test_output"

train_only_variables = ["yolo-v3/Conv_6",     # to train all layers, set this variable to None
                        "yolo-v3/Conv_14",
                        "yolo-v3/Conv_22",
                        "BatchNorm/beta",
                        "BatchNorm/gamma"]
gpu = "1"
learning_rate = 0.00025
decay_steps = 20000
decay_rate = 0.5
max_iter = 2100 * 30
start_iter = 1
num_threads = 2
#batch_size = 3
batch_size=1
#save_interval = 10000
save_interval = 5000
print_interval = 1
#print_interval = 200
summary_interval = 500

negative_ignore_thresh = 0.4
lambda_xy = 10
lambda_wh = 10
lambda_obj = 5
lambda_nonobj = 0.1
lambda_cls = 2
lambda_cs_conf = 5
lambda_alpha = 10
lambda_hwl = 1
weight_decay = 0.00005
random_crop = False
random_flip = False
random_color_shift = False
random_jitter = False
flip_chance = 0.5
color_shift_percentage = 0.5
jitter_percentage = 0.02
jitter_chance = 0.5

classes_to_consider = np.array(range(8))
class_map = {'Car': 0,
             'Van': 1,
             'Truck': 2,
             'Bus': 3,
             'Pedestrian': 4,
             'Traffic-cone': 5,
             'Bicyclist': 6,
             'Motorcyclist': 7}
assert num_classes == len(class_map), "num_classes must be equal to len(class_map)"
