import torch.nn as nn
from torchvision import models

'''
========================================================================
Model definition
========================================================================
'''


class TrajectoryImitationCNNFC(nn.Module):
    def __init__(self,
                 cnn_net=models.mobilenet_v2, pretrained=True,
                 pred_horizon=10):
        super(TrajectoryImitationCNNFC, self).__init__()
        # compressed to 3 channel
        self.compression_cnn_layer = nn.Conv2d(12, 3, 3, padding=1)
        self.cnn = cnn_net(pretrained=pretrained)
        for param in self.cnn.parameters():
            param.requires_grad = True

        self.feature_net_out_size = 1000
        self.pred_horizon = pred_horizon

        self.fc = nn.Sequential(
            nn.Linear(self.feature_net_out_size, 500),
            nn.Dropout(0.3),
            nn.Linear(500, 120),
            nn.Dropout(0.3),
            nn.Linear(120, self.pred_horizon * 4)
        )

    def forward(self, X):
        img = X
        out = self.compression_cnn_layer(img)
        out = self.cnn(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = out.view(out.size(0), self.pred_horizon, 4)
        return out


if __name__ == "__main__":
    # code snippet for model prining
    model = TrajectoryImitationCNNFC()

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
