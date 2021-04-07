"""Test training regime."""
import os
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn
from gandetect.models import DCTModel, LinearModel, SmallCNN
from gandetect.training import Training

from tests.utils import cifar, iris, test_dataset


class TestTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _common_kwargs(self):
        return {
            "model": SmallCNN(),
            "training_data": test_dataset(),
            "batch_size": 4,
            "patience": 1,
            "epochs": 1,
            "tensorboard": False,
            "save_path": TestTraining.tmp.name,
        }

    def test_training_runs(self):
        kwargs = self._common_kwargs()

        with Training(**kwargs) as training:
            training.train()

    def test_training_runs_multi_gpu(self):
        kwargs = self._common_kwargs()
        kwargs.update({"multi_gpu": True})

        with Training(**kwargs) as training:
            training.train()

    def test_loss_goes_down_and_acc_goes_up(self):
        kwargs = self._common_kwargs()
        kwargs.update({
            "training_data": cifar(),
            "batch_size": 128,
            "epochs": 3,
            "print_steps": -1,
        })

        with Training(**kwargs) as training:
            training.train()

            # check loss goes down
            for i in range(len(training.epoch_loss) - 1):
                self.assertGreater(
                    training.epoch_loss[i], training.epoch_loss[i+1])

            # check acc goes up
            for i in range(len(training.epoch_acc) - 1):
                self.assertLess(
                    training.epoch_acc[i], training.epoch_acc[i+1])

    def test_loss_goes_down_and_acc_goes_up_dct(self):
        kwargs = self._common_kwargs()
        kwargs.update({
            "training_data": cifar(),
            "model": DCTModel(model=SmallCNN()),
            "batch_size": 128,
            "learning_rate": 1e-3,
            "epochs": 5,
            "patience": 2,
            "print_steps": -1,
        })

        with Training(**kwargs) as training:
            training.train()

            # check loss goes down
            self.assertGreater(training.epoch_loss[0], training.epoch_loss[-1])

            # check acc goes up over training
            self.assertLess(training.epoch_acc[0], training.epoch_acc[-1])

    def test_linear_sep_works(self):
        kwargs = self._common_kwargs()
        kwargs.update({
            "training_data": iris(),
            "model": LinearModel(input_shape=(4)),
            "batch_size": 4,
            "epochs": 200,
            "patience": 5,
            "learning_rate": 1e-2,
        })
        with Training(**kwargs) as training:
            training.train()

            # check loss goes down
            for i in range(len(training.epoch_loss) - 1):
                self.assertGreater(
                    training.epoch_loss[i], training.epoch_loss[i+1])

            # check acc goes up
            for i in range(len(training.epoch_acc) - 1):
                if training.epoch_acc[i] == 100.:
                    continue

                self.assertLess(
                    training.epoch_acc[i], training.epoch_acc[i+1])

    def test_evaluate(self):
        kwargs = self._common_kwargs()
        training = Training(**kwargs)

        training.evaluate(test_dataset())

    def test_early_stopping_simple(self):
        epochs = 10
        kwargs = self._common_kwargs()
        kwargs.update({
            "epochs": epochs,
        })

        with Training(**kwargs) as training:
            training.train()

            self.assertLess(len(training.epoch_loss), epochs)
            self.assertGreater(len(training.epoch_loss), 1)

    def test_early_stopping_patience(self):
        epochs = 20
        kwargs = self._common_kwargs()
        kwargs.update({
            "epochs": epochs,
            "early_stopping_method": "patience",

        })
        with Training(**kwargs) as training:
            training.train()
            self.assertLess(len(training.epoch_loss), epochs)

            for param_group in training.optimizer.param_groups:
                self.assertLess(param_group["lr"], 1e-6)

    def test_tensorboard(self):
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = self._common_kwargs()
            kwargs.update({
                "tensorboard": True,
                "save_path": tmp,
            })

            with Training(**kwargs) as training:
                training.train()
                content = os.listdir(tmp)
                event_file_written = False
                checkpoint_written = False
                for con in content:
                    event_file_written |= "events.out.tfevents" in con
                    checkpoint_written |= con == "checkpoint.pth"

                self.assertEqual(len(content), 2)
                self.assertTrue(event_file_written, content)
                self.assertTrue(checkpoint_written)


if __name__ == "__main__":
    unittest.main()
