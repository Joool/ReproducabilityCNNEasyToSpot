"""Test the different transforamtions used."""
import unittest

import numpy as np
import torch
from gandetect.transforms import (_GAUSSIAN_BLUR_TRANSFORM, _JPEG_TRANSFORM,
                                  GAUSSIAN_BLUR_TRANSFORM, JPEG_TRANSFORM,
                                  NO_AUGMENT_TRANSFORM, TEST_TRANSFORM,
                                  _blur_jpeg, blur_jpeg)
from scipy import fft
from torchvision import transforms

from tests.utils import test_dataset


class TestTransformsInternal(unittest.TestCase):
    def _check(self, trans, min_val, max_val):
        data = test_dataset(transforms=transforms.Compose(
            [*trans, transforms.ToTensor()]))
        data_no_augment = test_dataset(
            transforms=transforms.Compose([transforms.ToTensor()]))

        diff = 0
        for (a, _), (b, _) in zip(data, data_no_augment):
            if not torch.allclose(a, b):
                diff += 1
            self.assertEqual(a.shape, (3, 256, 256))

        self.assertTrue(diff < int(max_val * len(data))
                        and diff > int(min_val * len(data)))

    def test_gaussian_blur_internal(self):
        self._check(_GAUSSIAN_BLUR_TRANSFORM, .35, .65)

    def test_jpeg_internal(self):
        self._check(_JPEG_TRANSFORM, .35, .65)

    def test_blur_jpeg_internal(self):
        self._check(_blur_jpeg(.3), .05, .45)


class TestTransforms(unittest.TestCase):
    def _check_training(self, trans, data_augmentation, dct=False):
        trans = trans.transforms

        # check correct length
        length = 4
        if data_augmentation:
            length += 1
        if dct:
            length -= 1

        self.assertEqual(len(trans), length)

        if data_augmentation is not None:
            self.assertIsInstance(trans[0], data_augmentation)

            # remove data_augmentation
            trans = trans[1:]

        # check correct training augmentations
        self.assertIsInstance(
            trans[0], transforms.RandomHorizontalFlip)
        self.assertIsInstance(trans[1], transforms.RandomCrop)

        if dct:
            self.assertIsInstance(trans[2], transforms.Lambda)
        else:
            self.assertIsInstance(trans[2], transforms.ToTensor)
            self.assertIsInstance(trans[3], transforms.Normalize)

    def test_gaussian_blur(self):
        self._check_training(GAUSSIAN_BLUR_TRANSFORM, transforms.RandomApply)

    def test_jpeg(self):
        self._check_training(JPEG_TRANSFORM, transforms.Lambda)

    def test_blur_jpeg(self):
        self._check_training(blur_jpeg(.3), transforms.Lambda)

    def test_no_aug(self):
        self._check_training(NO_AUGMENT_TRANSFORM, None)

    def test_test(self):
        trans = TEST_TRANSFORM.transforms
        self.assertEqual(len(trans), 3)

        self.assertIsInstance(trans[0], transforms.CenterCrop)
        self.assertIsInstance(trans[1], transforms.ToTensor)
        self.assertIsInstance(trans[2], transforms.Normalize)

    def test_blur_jpeg_dct(self):
        self._check_training(blur_jpeg(.3, dct=True),
                             transforms.Lambda, dct=True)


if __name__ == "__main__":
    unittest.main()
