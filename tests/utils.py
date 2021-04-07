"""This file contains utility functions for testing"""
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from gandetect.dataloader import ImageLabelDataset, image_paths_with_labels
from gandetect.transforms import IMAGE_SIZE, NO_AUGMENT_TRANSFORM
from PIL import Image
from sklearn.datasets import load_iris

DATA_PATH = f"{Path(__file__).parent.absolute()}/dummy"
FAKE_PATH = f"{DATA_PATH}/fake"
REAL_PATH = f"{DATA_PATH}/real"


def cifar():
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC),
            torchvision.transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform)
    trainset = list(filter(lambda x: x[1] == 0 or x[1] == 1, trainset))

    return trainset


def iris():
    # load and filter non seperable class
    X, y = list(
        zip(*filter(lambda x: x[1] != 1, zip(*load_iris(return_X_y=True)))))
    y = np.asarray(y)
    y[np.where(y == 2)] = 1
#
#     # pre shuffle data
#     data = list(zip(X, y))
#     np.random.shuffle(data)
#     X, y = list(zip(*data))

    X = torch.as_tensor(X, dtype=torch.float)
    y = torch.as_tensor(y, dtype=torch.int)
    data = torch.utils.data.TensorDataset(X, y)
    return data


def test_dataset(transforms=NO_AUGMENT_TRANSFORM):
    real = image_paths_with_labels(REAL_PATH, 0)
    fake = image_paths_with_labels(FAKE_PATH, 1)

    return ImageLabelDataset([real, fake], transformations=transforms)


def test_loader():
    dataset = test_dataset()

    return torch.utils.data.DataLoader(dataset, batch_size=8)
