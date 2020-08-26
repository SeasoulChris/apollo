import numpy as np
import torch
from torch import nn

from fueling.perception.pointpillars.second.pytorch.models.voxelnet import (
    register_voxelnet, VoxelNet, LossNormType)
from fueling.perception.pointpillars.second.pytorch.models import rpn


class SmallObjectHead(nn.Module):
    def __init__(self, num_filters, num_class, num_anchor_per_loc,
                 box_code_size, num_direction_bins, use_direction_classifier,
                 encode_background_as_zeros):
        super().__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._num_direction_bins = num_direction_bins
        self._num_class = num_class
        self._use_direction_classifier = use_direction_classifier
        self._box_code_size = box_code_size
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)

        self.net = nn.Sequential(
            nn.Conv2d(num_filters, 64, 3, bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        final_num_filters = 64
        self.conv_cls = nn.Conv2d(final_num_filters, num_cls, 1)
        self.conv_box = nn.Conv2d(final_num_filters,
                                  num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                final_num_filters, num_anchor_per_loc * num_direction_bins, 1)

    def forward(self, x):
        x = self.net(x)
        batch_size = x.shape[0]
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        # [N, C, y(H), x(W)]
        C, H, W = box_preds.shape[1:]
        box_preds = box_preds.view(-1, self._num_anchor_per_loc,
                                   self._box_code_size, H, W).permute(
                                       0, 1, 3, 4, 2).contiguous()
        cls_preds = cls_preds.view(-1, self._num_anchor_per_loc,
                                   self._num_class, H, W).permute(
                                       0, 1, 3, 4, 2).contiguous()
        ret_dict = {
            "box_preds": box_preds.view(batch_size, -1, self._box_code_size),
            "cls_preds": cls_preds.view(batch_size, -1, self._num_class),
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.view(
                -1, self._num_anchor_per_loc, self._num_direction_bins, H,
                W).permute(0, 1, 3, 4, 2).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds.view(batch_size, -1, self._num_direction_bins)
        return ret_dict


class DefaultHead(nn.Module):
    def __init__(self, num_filters, num_class, num_anchor_per_loc,
                 box_code_size, num_direction_bins, use_direction_classifier,
                 encode_background_as_zeros):
        super().__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._num_direction_bins = num_direction_bins
        self._num_class = num_class
        self._use_direction_classifier = use_direction_classifier
        self._box_code_size = box_code_size
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)

        final_num_filters = num_filters
        self.conv_cls = nn.Conv2d(final_num_filters, num_cls, 1)
        self.conv_box = nn.Conv2d(final_num_filters,
                                  num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                final_num_filters, num_anchor_per_loc * num_direction_bins, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        # [N, C, y(H), x(W)]
        C, H, W = box_preds.shape[1:]
        box_preds = box_preds.view(-1, self._num_anchor_per_loc,
                                   self._box_code_size, H, W).permute(
                                       0, 1, 3, 4, 2).contiguous()
        cls_preds = cls_preds.view(-1, self._num_anchor_per_loc,
                                   self._num_class, H, W).permute(
                                       0, 1, 3, 4, 2).contiguous()
        ret_dict = {
            "box_preds": box_preds.view(batch_size, -1, self._box_code_size),
            "cls_preds": cls_preds.view(batch_size, -1, self._num_class),
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.view(
                -1, self._num_anchor_per_loc, self._num_direction_bins, H,
                W).permute(0, 1, 3, 4, 2).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds.view(batch_size, -1, self._num_direction_bins)
        return ret_dict


@register_voxelnet
class VoxelNetNuscenesMultiHead(VoxelNet):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        assert self._num_class == 10
        assert isinstance(self.rpn, rpn.RPNNoHead)
        self.small_classes = ["pedestrian", "traffic_cone", "bicycle", "motorcycle", "barrier"]
        self.large_classes = ["car", "truck", "trailer", "bus", "construction_vehicle"]
        small_num_anchor_loc = sum(
            [self.target_assigner.num_anchors_per_location_class(c) for c in self.small_classes])
        large_num_anchor_loc = sum(
            [self.target_assigner.num_anchors_per_location_class(c) for c in self.large_classes])
        self.small_head = SmallObjectHead(
            num_filters=self.rpn._num_filters[0],
            num_class=self._num_class,
            num_anchor_per_loc=small_num_anchor_loc,
            encode_background_as_zeros=self._encode_background_as_zeros,
            use_direction_classifier=self._use_direction_classifier,
            box_code_size=self._box_coder.code_size,
            num_direction_bins=self._num_direction_bins,
        )
        self.large_head = DefaultHead(
            num_filters=np.sum(self.rpn._num_upsample_filters),
            num_class=self._num_class,
            num_anchor_per_loc=large_num_anchor_loc,
            encode_background_as_zeros=self._encode_background_as_zeros,
            use_direction_classifier=self._use_direction_classifier,
            box_code_size=self._box_coder.code_size,
            num_direction_bins=self._num_direction_bins,
        )

    def network_forward(self, voxels, num_points, coors, batch_size):
        self.start_timer("voxel_feature_extractor")
        voxel_features = self.voxel_feature_extractor(voxels, num_points,
                                                      coors)
        self.end_timer("voxel_feature_extractor")

        self.start_timer("middle forward")
        spatial_features = self.middle_feature_extractor(
            voxel_features, coors, batch_size)
        self.end_timer("middle forward")
        self.start_timer("rpn forward")
        rpn_out = self.rpn(spatial_features)
        r1 = rpn_out["stage0"]
        _, _, H, W = r1.shape
        cropsize40x40 = np.round(H * 0.1).astype(np.int64)
        r1 = r1[:, :, cropsize40x40:-cropsize40x40, cropsize40x40:-cropsize40x40]
        small = self.small_head(r1)
        large = self.large_head(rpn_out["out"])
        self.end_timer("rpn forward")
        # concated preds MUST match order in class_settings in config.
        res = {
            "box_preds": torch.cat([large["box_preds"], small["box_preds"]], dim=1),
            "cls_preds": torch.cat([large["cls_preds"], small["cls_preds"]], dim=1),
        }
        if self._use_direction_classifier:
            res["dir_cls_preds"] = torch.cat(
                [large["dir_cls_preds"], small["dir_cls_preds"]], dim=1)
        return res


@register_voxelnet
class ApolloVoxelNetNuscenesMultiHead(VoxelNet):
    def __init__(self,
                 output_shape,
                 num_class=2,
                 num_input_features=4,
                 vfe_class_name="VoxelFeatureExtractor",
                 vfe_num_filters=[32, 128],
                 with_distance=False,
                 middle_class_name="SparseMiddleExtractor",
                 middle_num_input_features=-1,
                 middle_num_filters_d1=[64],
                 middle_num_filters_d2=[64, 64],
                 rpn_class_name="RPN",
                 rpn_num_input_features=-1,
                 rpn_layer_nums=[3, 5, 5],
                 rpn_layer_strides=[2, 2, 2],
                 rpn_num_filters=[128, 128, 256],
                 rpn_upsample_strides=[1, 2, 4],
                 rpn_num_upsample_filters=[256, 256, 256],
                 use_norm=True,
                 use_groupnorm=False,
                 num_groups=32,
                 use_direction_classifier=True,
                 use_sigmoid_score=False,
                 encode_background_as_zeros=True,
                 use_rotate_nms=True,
                 multiclass_nms=False,
                 nms_score_thresholds=None,
                 nms_pre_max_sizes=None,
                 nms_post_max_sizes=None,
                 nms_iou_thresholds=None,
                 target_assigner=None,
                 cls_loss_weight=1.0,
                 loc_loss_weight=1.0,
                 pos_cls_weight=1.0,
                 neg_cls_weight=1.0,
                 direction_loss_weight=1.0,
                 loss_norm_type=LossNormType.NormByNumPositives,
                 encode_rad_error_by_sin=False,
                 loc_loss_ftor=None,
                 cls_loss_ftor=None,
                 measure_time=False,
                 voxel_generator=None,
                 post_center_range=None,
                 dir_offset=0.0,
                 sin_error_factor=1.0,
                 nms_class_agnostic=False,
                 num_direction_bins=2,
                 direction_limit_offset=0,
                 name='voxelnet'):

        super().__init__(output_shape,
                         num_class,
                         num_input_features,
                         vfe_class_name,
                         vfe_num_filters,
                         with_distance,
                         middle_class_name,
                         middle_num_input_features,
                         middle_num_filters_d1,
                         middle_num_filters_d2,
                         rpn_class_name,
                         rpn_num_input_features,
                         rpn_layer_nums,
                         rpn_layer_strides,
                         rpn_num_filters,
                         rpn_upsample_strides,
                         rpn_num_upsample_filters,
                         use_norm,
                         use_groupnorm,
                         num_groups,
                         use_direction_classifier,
                         use_sigmoid_score,
                         encode_background_as_zeros,
                         use_rotate_nms,
                         multiclass_nms,
                         nms_score_thresholds,
                         nms_pre_max_sizes,
                         nms_post_max_sizes,
                         nms_iou_thresholds,
                         target_assigner,
                         cls_loss_weight,
                         loc_loss_weight,
                         pos_cls_weight,
                         neg_cls_weight,
                         direction_loss_weight,
                         loss_norm_type,
                         encode_rad_error_by_sin,
                         loc_loss_ftor,
                         cls_loss_ftor,
                         measure_time,
                         voxel_generator,
                         post_center_range,
                         dir_offset,
                         sin_error_factor,
                         nms_class_agnostic,
                         num_direction_bins,
                         direction_limit_offset,
                         name)
        assert self._num_class == 10
        assert isinstance(self.rpn, rpn.RPNNoHead)
        self.small_classes = ["pedestrian", "traffic_cone", "bicycle", "motorcycle", "barrier"]
        self.large_classes = ["car", "truck", "trailer", "bus", "construction_vehicle"]
        small_num_anchor_loc = sum(
            [self.target_assigner.num_anchors_per_location_class(c) for c in self.small_classes])
        large_num_anchor_loc = sum(
            [self.target_assigner.num_anchors_per_location_class(c) for c in self.large_classes])
        small_head = SmallObjectHead(
            num_filters=self.rpn._num_filters[0],
            num_class=self._num_class,
            num_anchor_per_loc=small_num_anchor_loc,
            encode_background_as_zeros=self._encode_background_as_zeros,
            use_direction_classifier=self._use_direction_classifier,
            box_code_size=self._box_coder.code_size,
            num_direction_bins=self._num_direction_bins,
        )
        large_head = DefaultHead(
            num_filters=np.sum(self.rpn._num_upsample_filters),
            num_class=self._num_class,
            num_anchor_per_loc=large_num_anchor_loc,
            encode_background_as_zeros=self._encode_background_as_zeros,
            use_direction_classifier=self._use_direction_classifier,
            box_code_size=self._box_coder.code_size,
            num_direction_bins=self._num_direction_bins,
        )

        self.rpn = rpn.RPNMultiHead(
            small_head=small_head,
            large_head=large_head,
            use_norm=True,
            num_class=num_class,
            layer_nums=rpn_layer_nums,
            layer_strides=rpn_layer_strides,
            num_filters=rpn_num_filters,
            upsample_strides=rpn_upsample_strides,
            num_upsample_filters=rpn_num_upsample_filters,
            num_input_features=rpn_num_input_features,
            num_anchor_per_loc=self.target_assigner.num_anchors_per_location,
            encode_background_as_zeros=encode_background_as_zeros,
            use_direction_classifier=use_direction_classifier,
            use_groupnorm=use_groupnorm,
            num_groups=num_groups,
            box_code_size=self.target_assigner.box_coder.code_size,
            num_direction_bins=self._num_direction_bins)
