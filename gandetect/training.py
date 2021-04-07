import multiprocessing
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from gandetect import LOG


class Training(object):
    """Training wrapper for training models. Supports early stopping, saving the current best model and loading it on halt.."""

    def __init__(self,
                 model,
                 training_data,
                 epochs=50,
                 batch_size=128,
                 optimizer=torch.optim.Adam,
                 learning_rate=1e-4,
                 loss=torch.nn.BCEWithLogitsLoss,
                 print_steps=2_000,
                 use_cuda=True,
                 multi_gpu=False,
                 num_workers=multiprocessing.cpu_count(),
                 patience=5,
                 early_stopping_method="simple",
                 weight_decay=0,
                 tensorboard=True,
                 save_logits_histograms=False,
                 save_path="experiments"):
        """
        Args:
            model (gandetect.model): The model to train.
            training_data (torch.data.Dataset): A pytorch dataset holding the training data.
            epochs (int): Epochs to train for.
            batch_size (int): Batch size to use.
            optimizer (torch.optim.Optimizer): An instance of an optimizer class.
            loss (torch.nn.Loss): An instance of a loss class.
            print_steps (int): Print progress every x iterations.
            use_cuda (bool): Use GPU backend if available?
            multi_gpu (bool): Use multiple GPU if available?
            num_workers (int): Worker threads used for loading data.
            patience (int): Patience for early stopping.
            early_stopping_method (str): Method to use for ealry stopping:
                - simple (default): Stop if validation criteria does not improve after patience epochs.
                - patience: The learning rate gets lowered by 10x when the validation accuracy does not improve for patience epochs. 
                Terminate when learning rate reaches 10e-6.
            weight_decay (float): L2 regularization parameter.
            tensorboard (bool): Save results using tensorboard?
            save_logits_histograms (bool): Save histograms of the logits layer.
            save_path (str): Path to save tensorboard logs and model to.
        """
        super().__init__()

        if use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.multi_gpu = multi_gpu
        else:
            self.device = torch.device("cpu")

        # general parameters
        if multi_gpu and (torch.cuda.device_count() > 1):
            LOG.info(f"Training on {torch.cuda.device_count()} GPUs!")
            model = torch.nn.DataParallel(model)
            batch_size *= torch.cuda.device_count()

        self.model = model.to(self.device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers

        # setup model directory and logging
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.model_path = f"{self.save_path}/checkpoint.pth"

        self.tensorboard = tensorboard
        self.save_logits_histograms = save_logits_histograms
        if self.tensorboard:
            self.writer = SummaryWriter(self.save_path)

        # early stopping
        self.patience = patience
        self.best_eval = None
        self.counter = 0

        # setup training
        if len(training_data) == 0:
            raise ValueError("Provided empty dataset!")
        split = int(len(training_data) * 0.9)
        self.train_data, self.val_data = torch.utils.data.random_split(
            training_data, [split, len(training_data) - split])

        LOG.info(f"Using {len(self.val_data)} images for validation!")

        # setup training loader
        self.train_loader = torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        # setup optimzer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_fn = optimizer
        self.early_stopping_method = early_stopping_method
        self._init_optimizer()

        self.loss = loss()
        self.print_steps = print_steps

        # setup statistics
        self.epoch_loss = None
        self.epoch_acc = None

    def _init_optimizer(self):
        self.optimizer = self.optimizer_fn(params=self.model.parameters(),
                                           lr=self.learning_rate, weight_decay=self.weight_decay)

    def train(self):
        """Train the given model until either max epochs is reached or early stopping criteria is fullfilled."""
        self.epoch_loss = list()
        self.epoch_acc = list()
        self.model = self.model.train()

        for epoch in range(self.epochs):
            running_loss = 0.0
            epoch_loss = 0.0

            for i, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                # predict and optimize
                logits = self.model(x)
                loss = self.loss(logits.view(-1), y.double())
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                epoch_loss += loss.item()

                if i % self.print_steps == self.print_steps - 1:
                    print(
                        f"Epoch {epoch+1:03d} [{i+1: 5d}]: Running {running_loss / self.print_steps: .6f}, Current {loss.item(): .6f}\r", end="")
                    running_loss = 0.0
                    if self.save_logits_histograms:
                        self.writer.add_histogram(
                            "Train/LogitsDist", logits, (epoch * 1_000) + i)

            epoch_loss = epoch_loss / i
            LOG.info(
                f"Epoch {epoch+1:03d}: Loss: {epoch_loss}")
            self.epoch_loss.append(epoch_loss)

            acc = self.evaluate(self.val_data, current_epoch=epoch)
            self.epoch_acc.append(acc)

            if self.tensorboard:
                self.writer.add_scalar(
                    "Loss/train", epoch_loss, epoch)
                self.writer.add_scalar(
                    "Acc/val", acc, epoch)

            if self._early_stopping(acc):
                LOG.info(
                    f"Early Stopping after {epoch+1} epoch(s): best accuracy: {self.best_eval:.2f}; current: {acc:.2f}!")
                self.load_model()
                return

    def _early_stopping(self, eval_criteria):
        if self.best_eval is None or self.best_eval < eval_criteria:
            self.best_eval = eval_criteria
            self.counter = 0
            self.save_model()

        if self.patience <= self.counter:
            if self.early_stopping_method == "simple":
                return True
            elif self.early_stopping_method == "patience":
                # modify learning rate
                self.counter = 0

                for param_group in self.optimizer.param_groups:
                    param_group["lr"] /= 10

                    # if we reach abort criteria
                    if param_group["lr"] < 1e-6:
                        return True

                    LOG.info(
                        f"Reducing learning rate to {param_group['lr']:e}!")

        self.counter += 1

        return False

    def save_model(self):
        """Save the model"""
        torch.save(self.model, self.model_path)

    def load_model(self):
        """Load model"""
        torch.load(self.model_path)

    def evaluate(self, test, current_epoch=None):
        """Evaluate the model on the test data. Measures accuracy."""
        test_loader = torch.utils.data.DataLoader(
            test, batch_size=self.batch_size, num_workers=self.num_workers)

        total = 0
        correct = 0

        with torch.no_grad():
            self.model = self.model.eval()
            for i, (x, y) in enumerate(test_loader):
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                predicted = (torch.sigmoid(outputs) > 0.5).int()

                # save logits dist during training
                if current_epoch is not None and self.tensorboard and self.save_logits_histograms:
                    self.writer.add_histogram("Test/LogitsDist", outputs,
                                              (current_epoch * 1_000) + i)

                    self.model = self.model.train()
                    self.writer.add_histogram("Test/LogitsDistInTrainMode", self.model(x),
                                              (current_epoch * 1_000) + i)
                    self.model = self.model.eval()

                total += y.size(0)
                correct += (predicted.t() == y).sum().cpu().item()
            self.model = self.model.train()

        return correct / total * 100.

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        if self.tensorboard:
            self.writer.close()
