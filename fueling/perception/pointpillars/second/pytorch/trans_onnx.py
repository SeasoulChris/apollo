import os
import sys
import torch
from google.protobuf import text_format
from fueling.perception.pointpillars.second.protos import pipeline_pb2
from fueling.perception.pointpillars.second.pytorch.train import build_network


def trans_onnx(config_path, ckpt_path, save_onnx_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)

    model_cfg = config.model.second

    net = build_network(model_cfg, measure_time=False, export_onnx=True).to(device)
    net.load_state_dict(torch.load(ckpt_path))

    voxels = torch.ones([30000, 60, 5], dtype=torch.float32, device=device)
    num_points = torch.ones([30000], dtype=torch.float32, device=device)
    coors = torch.ones([30000, 4], dtype=torch.float32, device=device)

    example1 = (voxels, num_points, coors)

    spatial_features = torch.ones([1, 64, 400, 400], dtype=torch.float32, device=device)

    example2 = (spatial_features,)

    pfe_path = os.path.join(save_onnx_path, "pfe.onnx")
    rpn_path = os.path.join(save_onnx_path, "rpn.onnx")
    if not os.path.exists(save_onnx_path):
        os.makedirs(save_onnx_path)

    torch.onnx.export(net.voxel_feature_extractor, example1, pfe_path, verbose=False)
    torch.onnx.export(net.rpn, example2, rpn_path, verbose=False)


if __name__ == '__main__':

    config_path = sys.argv[1]
    ckpt_path = sys.argv[2]
    save_onnx_path = sys.argv[3]
    trans_onnx(config_path, ckpt_path, save_onnx_path)
