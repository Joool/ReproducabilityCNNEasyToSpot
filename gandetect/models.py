"""This module offers differnt models for classifying data. All expect 256x256 inputs."""
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as dct
from scipy.fft import dct
from torchvision import models

from gandetect.transforms import IMAGE_SIZE

# =========================================================
# VGG
# =========================================================


class VGG(nn.Module):
    """VGG convinience wrapper.
    """

    def __init__(self, vgg_type, pretrained=True):
        super().__init__()
        if vgg_type == "vgg11":
            vgg = models.vgg11(pretrained=pretrained)
        elif vgg_type == "vgg11_bn":
            vgg = models.vgg11_bn(pretrained=pretrained)
        else:
            raise ValueError("Did not specify a correct vgg type!")

        self.vgg = vgg.train()

        # replace fully connected layer
        self.vgg.classifier[6] = nn.Linear(
            self.vgg.classifier[6].in_features, 1)
        torch.nn.init.normal_(self.vgg.classifier[6].weight.data)

    def forward(self, x):
        x = self.vgg(x)
        return x

# =========================================================
# ResNet
# =========================================================


class ResNet(nn.Module):
    """ResNet convinience wrapper.
    """

    def __init__(self, resnet_type, pretrained=True):
        super().__init__()
        if resnet_type == "resnet18":
            resnet = models.resnet18(pretrained=pretrained)
        elif resnet_type == "resnet34":
            resnet = models.resnet34(pretrained=pretrained)
        elif resnet_type == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
        elif resnet_type == "resnet101":
            resnet = models.resnet101(pretrained=pretrained)
        elif resnet_type == "resnet152":
            resnet = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError("Did not specify a correct resnet type!")

        self.resnet = resnet.train()

        # replace fully connected layer
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)
        torch.nn.init.normal_(self.resnet.fc.weight.data)

    def forward(self, x):
        x = self.resnet(x)
        return x

# =========================================================
# CNN
# =========================================================


class SmallCNN(nn.Module):
    """A simple small CNN as baseline or testing."""

    def __init__(self):
        super().__init__()

        # 224 x 224
        self.conv_1 = nn.Conv2d(3, 8, 3, padding=1)

        # 64 x 64
        self.pool_1 = nn.MaxPool2d(4, 4)
        self.conv_2 = nn.Conv2d(8, 16, 3, padding=1)

        # 16 x 16
        self.pool_2 = nn.MaxPool2d(4, 4)
        self.conv_3 = nn.Conv2d(16, 32, 3, padding=1)

        # 4 x 4
        self.pool_3 = nn.MaxPool2d(4, 4)
        self.conv_4 = nn.Conv2d(32, 64, 3, padding=1)

        # 3 x 3
        self.pool_4 = nn.MaxPool2d(3, 3)
        self.conv_5 = nn.Conv2d(64, 128, 3, padding=1)

        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = self.pool_1(x)

        x = F.relu(self.conv_2(x))
        x = self.pool_2(x)

        x = F.relu(self.conv_3(x))
        x = self.pool_3(x)

        x = F.relu(self.conv_4(x))
        x = self.pool_4(x)

        x = F.relu(self.conv_5(x))

        x = x.view(-1, 128)
        x = self.fc(x)
        return x

# =========================================================
# MLP
# =========================================================


class MLP(nn.Module):
    """A simple linear model."""

    def __init__(self, input_shape=(3, IMAGE_SIZE, IMAGE_SIZE), hidden_size=512, dropout_rate=.7):
        super().__init__()
        self.input_shape = input_shape
        self.hidden_size = hidden_size

        self.hidden_one = nn.Linear(np.prod(self.input_shape), hidden_size)
        self.hidden_two = nn.Linear(hidden_size, hidden_size // 2)
        self.hidden_three = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(hidden_size // 4, 1)

    def forward(self, x):
        if x.shape[1:] != self.input_shape:
            raise ValueError(
                f"Provided incorrectly shaped input: {x.shape}; expected: {self.input_shape}!")

        x = x.flatten(start_dim=1)
        x = self.hidden_one(x)
        x = F.relu(x)
        x = self.hidden_two(x)
        x = F.relu(x)
        x = self.hidden_three(x)
        x = F.relu(x)
        x = self.dropout(x)

        return self.classifier(x)


# =========================================================
# Linear
# =========================================================


class LinearModel(nn.Module):
    """A simple linear model."""

    def __init__(self, input_shape=(3, IMAGE_SIZE, IMAGE_SIZE)):
        super().__init__()
        self.input_shape = input_shape
        self.fc = nn.Linear(np.prod(self.input_shape), 1)

    def forward(self, x):
        x = x.reshape([-1, np.prod(self.input_shape)])
        return self.fc(x)


# =========================================================
# DCT
# =========================================================
class DCTLayer(nn.Module):
    """DCT-2 preprocessing layer.

    Only supports square input."""

    def __init__(self, input_shape=IMAGE_SIZE):
        super().__init__()
        self.input_shape = input_shape
        I = np.eye(self.input_shape)
        I = dct(I, type=2, norm="ortho", axis=-1)

        self.register_buffer("dct", torch.as_tensor(I, dtype=torch.float32))

    def forward(self, x):
        # implements 2D-DCT-2 as precomputed matrix multiplication
        x = x.matmul(self.dct)

        x = x.transpose(-1, -2)
        x = x.matmul(self.dct)

        x = x.transpose(-1, -2)

        return x


class MinMaxScaler(nn.Module):
    """Layer for automatically learning min/max scaling.
    """

    def __init__(self, input_shape=(IMAGE_SIZE, IMAGE_SIZE)):
        super().__init__()

        self.input_shape = input_shape
        self.first = True
        self.register_buffer("min", torch.zeros(self.input_shape))
        self.register_buffer("max", torch.ones(self.input_shape))
        self.register_buffer("max_min", torch.ones(self.input_shape))

    def reset_parameters(self):
        self.min.zero_()
        self.max.one_()
        self.max_min.one_()
        self.first = True

    def forward(self, x):
        # check dims
        if x.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(x.dim()))
        if self.training:
            # batch stats
            x_min = torch.amin(x, dim=(0, 1))
            x_max = torch.amax(x, dim=(0, 1))

            if self.first:
                self.max = x_max
                self.min = x_min
                self.first = False

            else:
                # update min max with masking correect entries
                max_mask = torch.greater(x_max, self.max)
                self.max = (max_mask * x_max) + \
                    (torch.logical_not(max_mask) * self.max)

                min_mask = torch.less(x_min, self.min)
                self.min = (min_mask * x_min) + \
                    (torch.logical_not(min_mask) * self.min)

            self.max_min = self.max - self.min + 1e-13

        # scale batch
        x = (x - self.min) / self.max_min

        return x


def DCTModel(model=SmallCNN(), input_shape=IMAGE_SIZE):
    """Create a model with DCT preprocessing."""
    return nn.Sequential(OrderedDict([
        ("DCTPreprocessing", DCTLayer(input_shape=input_shape)),
        ("MinMaxScaling", MinMaxScaler()),
        ("BaseModel", model)
    ]))
