"""Utility functions for loading datasets."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from gandetect.transforms import NO_AUGMENT_TRANSFORM


class ImageDataset(Dataset):
    """Torch dataset to load a directory of images

    Args:
        directory (str): Path to the directory to load.
        transformations (torch.transforms): Transformations applied to the image on loading.
    Arguments:
        paths (str): Image paths.
        transforms (torch.transforms): Transformations applied to the image on loading.

    """

    def __init__(self, directory, transformations=NO_AUGMENT_TRANSFORM, **kwargs):
        super().__init__(**kwargs)
        if not Path(directory).exists():
            raise ValueError(f"Directory does not exist: {directory}!")

        # discover images from directory
        self.paths = image_paths(directory)
        if len(self.paths) == 0:
            raise ValueError(f"Directory did not contain images: {directory}!")

        self.transforms = transformations

    def __getitem__(self, index):
        path = self.paths[index]

        # open and apply transforms
        image = Image.open(path).convert('RGB')
        image = self.transforms(image)

        return image

    def __len__(self):
        return len(self.paths)


class ImageLabelDataset(Dataset):
    """Torch dataset to load a series of images from paths and labels.

    Args:
        paths_and_labels (str): Image paths and labels.
        transformations (torch.transforms): Transformations applied to the image on loading.
    Arguments:
        data (str): Image paths and labels.
        transforms (torch.transforms): Transformations applied to the image on loading.

    """

    def __init__(self, paths_and_labels, transformations=NO_AUGMENT_TRANSFORM, **kwargs):
        super().__init__(**kwargs)
        if len(paths_and_labels) == 0:
            raise ValueError("Did not supply data!")

        self.transforms = transformations

        # store data
        data = list()
        for data_pairs in paths_and_labels:
            data += data_pairs

        self.data = data

    def __getitem__(self, index):
        image, label = self.data[index]
        image = Image.open(image).convert('RGB')
        image = self.transforms(image)

        return image, label

    def __len__(self):
        return len(self.data)


def image_paths(dir_path):
    """Find all filepaths for images in dir_path."""
    return [str(path.absolute()) for path in sorted(_find_images(dir_path))]


def image_paths_with_labels(dir_path, label):
    """Find all filepaths for images in dir_path and attach a label."""
    paths = image_paths(dir_path)
    return list(zip(paths, [label] * len(paths)))


def image_paths_with_labels_from_path(dir_path):
    """Find all filepaths for images in dir_path and attach a label."""
    paths = image_paths(dir_path)

    def label_paths(path):
        label = 0 if "real" in path else 1
        return (path, label)

    return [label_paths(path) for path in paths]


def _find_images(data_path):
    """Returns all paths of images."""
    path = Path(data_path)

    paths = []
    paths += list(path.rglob("*.jpeg"))
    paths += list(path.rglob("*.png"))
    paths += list(path.rglob("*.jpg"))

    return paths


def load_data(path, transformations=NO_AUGMENT_TRANSFORM, limit=None):
    """Find all images in (sub)directory pointed to by path. Assign labels according to the path and return a train and test dataset."""
    data = image_paths_with_labels_from_path(path)
    real = list(filter(lambda x: x[1] == 0, data))
    fake = list(filter(lambda x: x[1] == 1, data))
    if limit is not None:
        real = real[:limit]
        fake = fake[:limit]
    amount = min(len(real), len(fake))

    real = real[:amount]
    fake = fake[:amount]

    split = int(0.85 * amount)

    real_train = real[:split]
    fake_train = fake[:split]

    real_test = real[split:]
    fake_test = fake[split:]

    train = ImageLabelDataset(
        [real_train, fake_train], transformations=transformations)
    test = ImageLabelDataset([real_test, fake_test],
                             transformations=transformations)

    return train, test


def load_data_from_multiple(path, transformations=NO_AUGMENT_TRANSFORM, limit_data=None, limit_directories=None):
    """Load from multiple directories and return the concat over all.
    """
    trains = list()
    tests = list()
    directories = list(sorted(Path(path).iterdir()))
    if limit_directories is not None:
        directories = directories[:limit_directories]

    for directory in directories:
        if not directory.is_dir():
            continue

        train, test = load_data(
            directory, transformations=transformations, limit=limit_data)
        trains.append(train)
        tests.append(test)

    train = torch.utils.data.ConcatDataset(trains)
    test = torch.utils.data.ConcatDataset(tests)
    return train, test
