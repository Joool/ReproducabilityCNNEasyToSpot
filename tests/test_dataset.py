"""Test the dataset loader"""
import unittest

import numpy as np
import torch
from gandetect.dataloader import (ImageDataset, ImageLabelDataset,
                                  image_paths_with_labels,
                                  image_paths_with_labels_from_path)
from gandetect.transforms import IMAGE_SIZE

from tests.utils import DATA_PATH, FAKE_PATH, REAL_PATH


class TestDataset(unittest.TestCase):

    def test_simple_dataset(self):
        dataset = ImageDataset(f"{DATA_PATH}/fake")

        for img in dataset:
            self.assertEqual(img.dtype, torch.float32)
            self.assertEqual(img.shape, torch.Size(
                [3, IMAGE_SIZE, IMAGE_SIZE]), img.shape)

    def test_invalid_path(self):
        with self.assertRaises(ValueError):
            dataset = ImageDataset("dwadwadawwda")

    def test_no_images_in_directory(self):
        with self.assertRaises(ValueError):
            dataset = ImageDataset("{DATA_PATH}/empty")


class TestDiscoveryFunctions(unittest.TestCase):

    def test_image_paths_with_labels(self):
        data = image_paths_with_labels(REAL_PATH, 0)

        self.assertEqual(data[0], (f"{REAL_PATH}/lsun_0.jpeg", 0))
        self.assertEqual(len(data), 99)

    def test_image_paths_with_labels_from_path(self):
        data = image_paths_with_labels_from_path(DATA_PATH)

        self.assertEqual(data[0], (f"{DATA_PATH}/fake/example_0.jpeg", 1))
        self.assertEqual(data[-1], (f"{DATA_PATH}/real/lsun_1000083.jpeg", 0))
        self.assertEqual(len(data), 198)


class TestDataloader(unittest.TestCase):

    def test_loading_two_directories(self):
        real = image_paths_with_labels(REAL_PATH, 0)
        fake = image_paths_with_labels(FAKE_PATH, 1)

        dataset = ImageLabelDataset([real, fake])

        # all data loaded
        self.assertEqual(len(dataset.data), 198)
        self.assertEqual(dataset.data[0][1], 0)
        self.assertEqual(dataset.data[-1][1], 1)


if __name__ == "__main__":
    unittest.main()
