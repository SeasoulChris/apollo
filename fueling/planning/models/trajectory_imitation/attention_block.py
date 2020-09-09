import torch
import torch.nn as nn


class SELayer(nn.Module):
    def __init__(self, channel, multi=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel * multi, bias=False),
                                nn.ReLU(),
                                nn.Linear(channel * multi, channel, bias=False),
                                nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, multi=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(nn.Conv2d(in_planes, in_planes * multi, 1, bias=False),
                                       nn.ReLU(),
                                       nn.Conv2d(in_planes * multi, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class ChannelSpatialSequentialBlock(nn.Module):
    def __init__(self, channel, multi=2):
        super(ChannelSpatialSequentialBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(channel, channel * multi, 3, padding=1),
                                   nn.BatchNorm2d(channel * multi),
                                   nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(channel * multi, channel * multi, 5, padding=2),
                                   nn.BatchNorm2d(channel * multi),
                                   nn.ReLU())

        self.conv3 = nn.Sequential(nn.Conv2d(channel * multi, channel, 3, padding=1),
                                   nn.BatchNorm2d(channel))

        self.ca = ChannelAttention(channel)
        self.sa = SpatialAttention()

        self.relu = nn.ReLU()

        self.compress = nn.Sequential(nn.Conv2d(channel, 3, 3, padding=1),
                                      nn.ReLU())

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        out += identity
        out = self.relu(out)

        out = self.compress(out)
        return out
