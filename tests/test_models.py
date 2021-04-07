"""Test cases for models."""
import unittest

import numpy as np
import torch
from gandetect.models import (MLP, DCTLayer, DCTModel, LinearModel,
                              MinMaxScaler, SmallCNN)
from gandetect.transforms import IMAGE_SIZE
from scipy import fft
from sklearn import preprocessing

from tests.utils import test_loader

MODELS = [
    ("Small CNN", SmallCNN()),
    ("DCT CNN", DCTModel()),
    ("MLP", MLP()),
    ("Linear model", LinearModel()),
    ("DCT Linear model", DCTModel(model=LinearModel())),
]


class TestModels(unittest.TestCase):
    def test_models_run(self):
        """Assert that the model runs."""
        loader = test_loader()
        for name, model in MODELS:
            with self.subTest(name):
                x, _ = next(iter(loader))
                model(x)

    def test_models_run_cuda(self):
        """Assert that the model runs."""
        if torch.cuda.is_available():
            loader = test_loader()
            for name, model in MODELS:
                with self.subTest(name):
                    x, _ = next(iter(loader))
                    model.cuda()(x.cuda())


class TestDCT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.default_rng(42)

    def test_dct_preserves_shape(self):
        dct = DCTLayer()

        x = TestDCT.rng.uniform(low=0., high=255., size=[
            64, 3, IMAGE_SIZE, IMAGE_SIZE]).astype(np.float32)
        x = torch.as_tensor(x)

        # assert shape the same
        self.assertEqual(x.shape, dct(x).shape)

    def test_dct_layer(self):
        dct = DCTLayer()

        for _ in range(10):
            x = TestDCT.rng.uniform(low=0., high=255., size=[
                64, 3, IMAGE_SIZE, IMAGE_SIZE]).astype(np.float32)
            x_impl = dct(torch.Tensor(x.copy()).float())

            x_ref = fft.dct(x, type=2, norm="ortho", axis=-1)
            x_ref = fft.dct(x_ref, type=2, norm="ortho", axis=-2)

            self.assertTrue(np.isclose(
                x_impl, x_ref, atol=1e-02).all(), f"{x_impl[0,0,0,:5]} {x_ref[0,0,0,:5]}")

    def test_dct_layer_cuda(self):
        dct = DCTLayer().cuda()

        x = np.random.normal(size=[12, 3, IMAGE_SIZE, IMAGE_SIZE])
        x_impl = dct(torch.Tensor(x.copy()).float().cuda())

        x_ref = fft.dct(x, type=2, norm="ortho", axis=-1)
        x_ref = fft.dct(x_ref, type=2, norm="ortho", axis=-2)

        self.assertTrue(np.isclose(x_impl.cpu(), x_ref, atol=1e-05).all())


class TestMinMaxScaler(unittest.TestCase):
    def test_minmax(self):
        scaler = MinMaxScaler(input_shape=(2, 2))
        x = torch.as_tensor(
            [[[2, 2], [-2, -2]], [[1, 3], [6, 2]], [[7, 7], [-7, -7]]], dtype=torch.float32)

        # channel dim
        x.unsqueeze_(1)

        # should not change anything yet
        scaler.eval()
        self.assertTrue(torch.equal(x, scaler(x)))

        # learn parameters
        scaler.train()

        scaler(x)
        self.assertTrue(torch.equal(
            torch.as_tensor(
                [[1, 2], [-7, -7]],
                dtype=torch.float32),
            scaler.min), scaler.min)

        self.assertTrue(torch.equal(
            torch.as_tensor(
                [[7, 7], [6, 2]],
                dtype=torch.float32),
            scaler.max), scaler.max)

        x = torch.as_tensor([[[[10, 10], [-10, -10]]]])
        res = scaler(x)

        self.assertTrue(torch.equal(
            torch.as_tensor(
                [[1, 2], [-10, -10]],
                dtype=torch.float32),
            scaler.min), scaler.min)

        self.assertTrue(torch.equal(
            torch.as_tensor(
                [[10, 10], [6, 2]],
                dtype=torch.float32),
            scaler.max), scaler.max)

        # test eval and train produce same result
        scaler.eval()
        self.assertTrue(torch.equal(
            res, scaler(x)
        ))

    def test_minmax_fuzz(self):
        scaler = MinMaxScaler(input_shape=(2, 2))
        scaler.train()

        for _ in range(100):
            x = torch.normal(0, 1, size=(32, 1, 2, 2))
            res = scaler(x)
            self.assertTrue(torch.less_equal(res, 1.).all())
            self.assertTrue(torch.greater_equal(res, 0.).all())


if __name__ == "__main__":
    unittest.main()
