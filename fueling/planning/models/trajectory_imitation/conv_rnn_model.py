import numpy as np
import torch
import torch.nn as nn
from torchvision import models


@torch.jit.script
def assign_ones(M_k_next: torch.Tensor, batch_iteration_num: torch.Tensor,
                arg_max_index: torch.Tensor, one_tensor: torch.Tensor):
    M_k_next[batch_iteration_num, arg_max_index] = one_tensor
    return M_k_next


class TrajectoryImitationConvRNN(nn.Module):
    def __init__(self, input_img_size,
                 cnn_net=models.mobilenet_v2, pretrained=True,
                 pred_horizon=10):
        super(TrajectoryImitationConvRNN, self).__init__()
        self.feature_compression_layer = nn.Conv2d(12, 3, 3, padding=1)

        self.feature_net = cnn_net(pretrained=pretrained)
        for param in self.feature_net.parameters():
            param.requires_grad = True

        self.feature_net_out_size = 1000
        self.pred_horizon = pred_horizon
        self.input_img_size_h = input_img_size[0]
        self.input_img_size_w = input_img_size[1]

        self.memory_encoder = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=121,
                      padding=10),  # size self.input_img_size_h to 100
            nn.ReLU()
        )

        self.memory_fc_layer = nn.Sequential(
            nn.Linear(100 * 100, 256),
            nn.ReLU()
        )

        self.input_fc_layer = nn.Sequential(
            nn.Linear(256 + 1000, 256),
            nn.ReLU()
        )

        self.input_deconv_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=2,
                               kernel_size=70, padding=10, stride=10),
            # size 16 * 16 to self.input_img_size_h,
            nn.ReLU()
        )

        self.output_conv_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=2, out_channels=1, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.output_fc_layers = nn.Sequential(
            nn.Linear(self.input_img_size_h * self.input_img_size_w, 4)
        )

    def forward(self, X):
        img_feature = X[0]
        batch_size = img_feature.size(0)
        M_B_k = torch.cat((X[1], X[2]), dim=1).to(img_feature.device)

        img_feature_encoding = self.feature_net(
            self.feature_compression_layer(img_feature))

        # initialize the result tensor with a redundant timestamp info
        # rather than empty because of a issue during TensorRT parsing onnx
        pred_pos_dists = torch.zeros(
            (batch_size, 1, 1,
             self.input_img_size_h, self.input_img_size_w),
            device=img_feature.device)
        pred_boxs = torch.zeros(
            (batch_size, 1, 1,
             self.input_img_size_h, self.input_img_size_w),
            device=img_feature.device)
        pred_points = torch.zeros(
            (batch_size, 1, 4), device=img_feature.device)

        for t in range(self.pred_horizon):
            P_B_k = self.memory_encoder(M_B_k)
            P_B_k = P_B_k.view(batch_size, -1)
            P_B_k = self.memory_fc_layer(P_B_k)

            F_P_B_k = torch.cat((img_feature_encoding, P_B_k), dim=1)
            F_P_B_k = self.input_fc_layer(F_P_B_k)
            F_P_B_k = F_P_B_k.view(batch_size, 1, 16, 16)
            F_P_B_k = self.input_deconv_layer(F_P_B_k)

            P_k = F_P_B_k[:, 0, :, :].clone()
            B_k = F_P_B_k[:, 1, :, :].clone()
            P_k = torch.softmax(
                P_k.view(batch_size, -1), dim=1).view(batch_size,
                                                      self.input_img_size_h,
                                                      self.input_img_size_w)
            B_k = torch.sigmoid(B_k)

            pred_pos_dists = torch.cat(
                (pred_pos_dists, P_k.unsqueeze(1).unsqueeze(1)), dim=1)
            pred_boxs = torch.cat(
                (pred_boxs, B_k.unsqueeze(1).unsqueeze(1)), dim=1)

            pred_point = self.output_conv_layer(F_P_B_k)
            pred_point = pred_point.view(batch_size, -1)
            pred_point = self.output_fc_layers(pred_point)
            pred_points = torch.cat(
                (pred_points, pred_point.clone().unsqueeze(1)), dim=1)

            arg_max_index = torch.argmax(
                F_P_B_k[:, 0, :, :].view(batch_size, -1), dim=1)
            M_k_next = M_B_k[:, 0, :, :].clone()
            M_k_next = M_k_next.view(batch_size, -1)
            batch_iteration_num = np.arange(batch_size)
            batch_iteration_num = torch.from_numpy(
                batch_iteration_num).to(img_feature.device)
            one_tensor = torch.ones((1), device=img_feature.device)
            M_k_next = assign_ones(
                M_k_next, batch_iteration_num, arg_max_index, one_tensor)
            M_k_next = M_k_next.view(batch_size, self.input_img_size_h,
                                     self.input_img_size_w)
            B_k_next = F_P_B_k[:, 1, :, :].clone()
            M_B_k = torch.stack((M_k_next, B_k_next), dim=1)

        # only return predicted points when exporting onnx or jit tracing
        # return pred_points[:, 1:, :]

        return (pred_points[:, 1:, :],
                pred_boxs[:, 1:, :, :, :],
                pred_pos_dists[:, 1:, :, :, :],
                M_B_k)


class TrajectoryImitationDeeperConvRNN(nn.Module):
    def __init__(self, input_img_size,
                 cnn_net=models.mobilenet_v2, pretrained=True,
                 pred_horizon=10):
        super(TrajectoryImitationDeeperConvRNN, self).__init__()
        self.feature_compression_layer = nn.Conv2d(12, 3, 3, padding=1)

        self.feature_net = cnn_net(pretrained=pretrained)
        for param in self.feature_net.parameters():
            param.requires_grad = True

        self.feature_net_out_size = 1000
        self.pred_horizon = pred_horizon
        self.input_img_size_h = input_img_size[0]
        self.input_img_size_w = input_img_size[1]

        self.memory_encoder = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=121,
                      padding=10),  # size self.input_img_size_h to 100
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=71,
                      padding=10),  # size 100 to 50
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=46,
                      padding=10),  # size 50 to 25
            nn.ReLU()
        )

        self.memory_fc_layer = nn.Sequential(
            nn.Linear(25 * 25, 256),
            nn.ReLU()
        )

        self.input_fc_layer = nn.Sequential(
            nn.Linear(256 + 1000, 256),
            nn.ReLU()
        )

        self.input_deconv_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=2,
                               kernel_size=55, padding=10, stride=1),
            # size 16 to 50,
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=2, out_channels=2,
                               kernel_size=71, padding=10, stride=1),
            # size 50 to 100,
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=2, out_channels=2,
                               kernel_size=121, padding=10, stride=1),
            # size 100 to self.input_img_size_h,
            nn.ReLU()
        )

        self.output_conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=121,
                      padding=10),  # size self.input_img_size_h to 100
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=71,
                      padding=10),  # size 100 to 50
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=46,
                      padding=10),  # size 50 to 25
            nn.ReLU()
        )

        self.output_fc_layers = nn.Sequential(
            nn.Linear(25 * 25, 4),
        )

    def forward(self, X):
        img_feature = X[0]
        batch_size = img_feature.size(0)
        M_B_k = torch.cat((X[1], X[2]), dim=1)

        img_feature_encoding = self.feature_net(
            self.feature_compression_layer(img_feature))

        # initialize the result tensor with a redundant timestamp info
        # rather than empty because of a issue during TensorRT parsing onnx
        pred_pos_dists = torch.zeros(
            (batch_size, 1, 1,
             self.input_img_size_h, self.input_img_size_w),
            device=img_feature.device)
        pred_boxs = torch.zeros(
            (batch_size, 1, 1,
             self.input_img_size_h, self.input_img_size_w),
            device=img_feature.device)
        pred_points = torch.zeros(
            (batch_size, 1, 4), device=img_feature.device)

        for t in range(self.pred_horizon):
            P_B_k = self.memory_encoder(M_B_k)
            P_B_k = P_B_k.view(batch_size, -1)
            P_B_k = self.memory_fc_layer(P_B_k)

            F_P_B_k = torch.cat((img_feature_encoding, P_B_k), dim=1)
            F_P_B_k = self.input_fc_layer(F_P_B_k)
            F_P_B_k = F_P_B_k.view(batch_size, 1, 16, 16)
            F_P_B_k = self.input_deconv_layer(F_P_B_k)

            P_k = F_P_B_k[:, 0, :, :].clone()
            B_k = F_P_B_k[:, 1, :, :].clone()
            P_k = torch.softmax(
                P_k.view(batch_size, -1), dim=1).view(batch_size,
                                                      self.input_img_size_h,
                                                      self.input_img_size_w)
            B_k = torch.sigmoid(B_k)

            pred_pos_dists = torch.cat(
                (pred_pos_dists, P_k.unsqueeze(1).unsqueeze(1)), dim=1)
            pred_boxs = torch.cat(
                (pred_boxs, B_k.unsqueeze(1).unsqueeze(1)), dim=1)

            pred_point = self.output_conv_layer(F_P_B_k)
            pred_point = pred_point.view(batch_size, -1)
            pred_point = self.output_fc_layers(pred_point)
            pred_points = torch.cat(
                (pred_points, pred_point.clone().unsqueeze(1)), dim=1)

            arg_max_index = torch.argmax(
                F_P_B_k[:, 0, :, :].view(batch_size, -1), dim=1)
            M_k_next = M_B_k[:, 0, :, :].clone()
            M_k_next = M_k_next.view(batch_size, -1)
            batch_iteration_num = np.arange(batch_size)
            batch_iteration_num = torch.from_numpy(
                batch_iteration_num).to(img_feature.device)
            one_tensor = torch.ones((1), device=img_feature.device)
            M_k_next = assign_ones(
                M_k_next, batch_iteration_num, arg_max_index, one_tensor)
            M_k_next = M_k_next.view(batch_size, self.input_img_size_h,
                                     self.input_img_size_w)
            B_k_next = F_P_B_k[:, 1, :, :].clone()
            M_B_k = torch.stack((M_k_next, B_k_next), dim=1)

        # only return predicted points when exporting onnx or jit tracing
        # return pred_points[:, 1:, :]

        return (pred_points[:, 1:, :],
                pred_boxs[:, 1:, :, :, :],
                pred_pos_dists[:, 1:, :, :, :],
                M_B_k)


class UnetDecoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(UnetDecoder, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv_relu(x1)
        return x1


class TrajectoryImitationConvRNNUnetResnet18v1(nn.Module):
    def __init__(self, input_img_size,
                 cnn_net=models.mobilenet_v2, pretrained=True,
                 pred_horizon=10):
        super(TrajectoryImitationConvRNNUnetResnet18v1, self).__init__()

        self.pred_horizon = pred_horizon
        self.input_img_size_h = input_img_size[0]
        self.input_img_size_w = input_img_size[1]

        self.base_model = models.resnet18(True)
        self.base_layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(
            nn.Conv2d(14, 64, kernel_size=(7, 7), stride=(
                2, 2), padding=(3, 3), bias=False),
            self.base_layers[1],
            self.base_layers[2])
        self.layer2 = nn.Sequential(*self.base_layers[3:5])
        self.layer3 = self.base_layers[5]
        self.decode2 = UnetDecoder(128, 128 + 64, 128)
        self.decode1 = UnetDecoder(128, 64 + 64, 64)
        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 2, kernel_size=3, padding=1, bias=False)
        )

        self.output_conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=121,
                      padding=10),  # size self.input_img_size_h to 100
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=71,
                      padding=10),  # size 100 to 50
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=46,
                      padding=10),  # size 50 to 25
            nn.ReLU()
        )

        self.output_fc_layers = nn.Sequential(
            nn.Linear(25 * 25, 4),
        )

    def forward(self, X):
        img_feature = X[0]
        batch_size = img_feature.size(0)
        M_B_k = torch.cat((X[1], X[2]), dim=1)

        # initialize the result tensor with a redundant timestamp info
        # rather than empty because of a issue during TensorRT parsing onnx
        pred_pos_dists = torch.zeros(
            (batch_size, 1, 1,
             self.input_img_size_h, self.input_img_size_w),
            device=img_feature.device)
        pred_boxs = torch.zeros(
            (batch_size, 1, 1,
             self.input_img_size_h, self.input_img_size_w),
            device=img_feature.device)
        pred_points = torch.zeros(
            (batch_size, 1, 4), device=img_feature.device)

        for t in range(self.pred_horizon):
            stacked_imgs = torch.cat((img_feature, M_B_k), dim=1)

            e1 = self.layer1(stacked_imgs)  # 64,100,100
            e2 = self.layer2(e1)  # 64,50,50
            e3 = self.layer3(e2)  # 128,25,25
            d2 = self.decode2(e3, e2)  # 128,50,50
            d1 = self.decode1(d2, e1)  # 64,100,100
            F_P_B_k = self.decode0(d1)  # 64,200,200

            P_k = F_P_B_k[:, 0, :, :].clone()
            B_k = F_P_B_k[:, 1, :, :].clone()
            P_k = torch.softmax(
                P_k.view(batch_size, -1), dim=1).view(batch_size,
                                                      self.input_img_size_h,
                                                      self.input_img_size_w)
            B_k = torch.sigmoid(B_k)

            pred_pos_dists = torch.cat(
                (pred_pos_dists, P_k.unsqueeze(1).unsqueeze(1)), dim=1)
            pred_boxs = torch.cat(
                (pred_boxs, B_k.unsqueeze(1).unsqueeze(1)), dim=1)

            pred_point = self.output_conv_layer(F_P_B_k)
            pred_point = pred_point.view(batch_size, -1)
            pred_point = self.output_fc_layers(pred_point)
            pred_points = torch.cat(
                (pred_points, pred_point.clone().unsqueeze(1)), dim=1)

            arg_max_index = torch.argmax(
                F_P_B_k[:, 0, :, :].view(batch_size, -1), dim=1)
            M_k_next = M_B_k[:, 0, :, :].clone()
            M_k_next = M_k_next.view(batch_size, -1)
            batch_iteration_num = np.arange(batch_size)
            batch_iteration_num = torch.from_numpy(
                batch_iteration_num).to(img_feature.device)
            one_tensor = torch.ones((1), device=img_feature.device)
            M_k_next = assign_ones(
                M_k_next, batch_iteration_num, arg_max_index, one_tensor)
            M_k_next = M_k_next.view(batch_size, self.input_img_size_h,
                                     self.input_img_size_w)
            B_k_next = F_P_B_k[:, 1, :, :].clone()
            M_B_k = torch.stack((M_k_next, B_k_next), dim=1)

        # only return predicted points when exporting onnx or jit tracing
        # return pred_points[:, 1:, :]

        return (pred_points[:, 1:, :],
                pred_boxs[:, 1:, :, :, :],
                pred_pos_dists[:, 1:, :, :, :],
                M_B_k)


class TrajectoryImitationConvRNNUnetResnet18v2(nn.Module):
    def __init__(self, input_img_size,
                 cnn_net=models.mobilenet_v2, pretrained=True,
                 pred_horizon=10):
        super(TrajectoryImitationConvRNNUnetResnet18v2, self).__init__()

        self.pred_horizon = pred_horizon
        self.input_img_size_h = input_img_size[0]
        self.input_img_size_w = input_img_size[1]

        self.base_model = models.resnet18(True)
        self.base_layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(
            nn.Conv2d(14, 64, kernel_size=(7, 7), stride=(
                2, 2), padding=(3, 3), bias=False),
            self.base_layers[1],
            self.base_layers[2])
        self.layer2 = nn.Sequential(*self.base_layers[3:5])
        self.layer3 = self.base_layers[5]
        self.layer4 = self.base_layers[6]
        self.layer5 = self.base_layers[7]
        self.layer6 = self.base_layers[8]
        self.decode2 = UnetDecoder(128, 128 + 64, 128)
        self.decode1 = UnetDecoder(128, 64 + 64, 64)
        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 2, kernel_size=3, padding=1, bias=False)
        )

        self.output_fc_layers = nn.Sequential(
            nn.Linear(512, 4),
        )

    def forward(self, X):
        img_feature = X[0]
        batch_size = img_feature.size(0)
        M_B_k = torch.cat((X[1], X[2]), dim=1)

        # initialize the result tensor with a redundant timestamp info
        # rather than empty because of a issue during TensorRT parsing onnx
        pred_pos_dists = torch.zeros(
            (batch_size, 1, 1,
             self.input_img_size_h, self.input_img_size_w),
            device=img_feature.device)
        pred_boxs = torch.zeros(
            (batch_size, 1, 1,
             self.input_img_size_h, self.input_img_size_w),
            device=img_feature.device)
        pred_points = torch.zeros(
            (batch_size, 1, 4), device=img_feature.device)

        for t in range(self.pred_horizon):
            stacked_imgs = torch.cat((img_feature, M_B_k), dim=1)

            e1 = self.layer1(stacked_imgs)  # 64,100,100
            e2 = self.layer2(e1)  # 64,50,50
            e3 = self.layer3(e2)  # 128,25,25

            output_e1 = e3.clone()
            output_e2 = self.layer4(output_e1)
            output_e3 = self.layer5(output_e2)
            output_e4 = self.layer6(output_e3)
            output_e4 = output_e4.view(batch_size, -1)
            pred_point = self.output_fc_layers(output_e4)
            pred_points = torch.cat(
                (pred_points, pred_point.clone().unsqueeze(1)), dim=1)

            d2 = self.decode2(e3, e2)  # 128,50,50
            d1 = self.decode1(d2, e1)  # 64,100,100
            P_B_k = self.decode0(d1)  # 2,200,200

            P_k = P_B_k[:, 0, :, :].clone()
            B_k = P_B_k[:, 1, :, :].clone()
            P_k = torch.softmax(
                P_k.view(batch_size, -1), dim=1).view(batch_size,
                                                      self.input_img_size_h,
                                                      self.input_img_size_w)
            B_k = torch.sigmoid(B_k)

            pred_pos_dists = torch.cat(
                (pred_pos_dists, P_k.clone().unsqueeze(1).unsqueeze(1)), dim=1)
            pred_boxs = torch.cat(
                (pred_boxs, B_k.clone().unsqueeze(1).unsqueeze(1)), dim=1)

            arg_max_index = torch.argmax(
                P_k.view(batch_size, -1), dim=1)
            M_k_next = M_B_k[:, 0, :, :].clone()
            M_k_next = M_k_next.view(batch_size, -1)
            batch_iteration_num = np.arange(batch_size)
            batch_iteration_num = torch.from_numpy(
                batch_iteration_num).to(img_feature.device)
            one_tensor = torch.ones((1), device=img_feature.device)
            M_k_next = assign_ones(
                M_k_next, batch_iteration_num, arg_max_index, one_tensor)
            M_k_next = M_k_next.view(batch_size, self.input_img_size_h,
                                     self.input_img_size_w)
            B_k_next = B_k.clone()
            M_B_k = torch.stack((M_k_next, B_k_next), dim=1)

        # only return predicted points when exporting onnx or jit tracing
        # return pred_points[:, 1:, :]

        return (pred_points[:, 1:, :],
                pred_boxs[:, 1:, :, :, :],
                pred_pos_dists[:, 1:, :, :, :],
                M_B_k)


if __name__ == "__main__":
    # code snippet for model prining
    model = TrajectoryImitationConvRNN([200, 200])

    total_param = 0
    for name, param in model.named_parameters():
        print(name)
        print(param.data.shape)
        print(param.data.view(-1).shape[0])
        total_param += param.data.view(-1).shape[0]
    print(total_param)

    for index, (name, child) in enumerate(model.named_children()):
        print(index)
        print(name)
        print(child)
