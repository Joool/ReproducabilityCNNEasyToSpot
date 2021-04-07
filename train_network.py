import argparse
import logging
import os
import sys

import torch

from gandetect.dataloader import (ImageLabelDataset, image_paths_with_labels,
                                  load_data_from_multiple)
from gandetect.models import MLP, VGG, DCTModel, LinearModel, ResNet
from gandetect.training import Training
from gandetect.transforms import (GAUSSIAN_BLUR_TRANSFORM,
                                  GAUSSIAN_BLUR_TRANSFORM_DCT, JPEG_TRANSFORM,
                                  JPEG_TRANSFORM_DCT, NO_AUGMENT_TRANSFORM,
                                  NO_AUGMENT_TRANSFORM_DCT, blur_jpeg,
                                  blur_then_jpeg)
from gandetect.utils import set_seed


def setup_logging():
    log_formatter = logging.Formatter(
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()

    file_handler = logging.FileHandler("train_log.log")
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    root_logger.setLevel(logging.INFO)


EVAL_MODELS = {
    "dct_resnet18": (DCTModel(model=ResNet("resnet18", pretrained=False)), {"learning_rate": 1e-3, "multi_gpu": False}),
    "dct_resnet50": (DCTModel(model=ResNet("resnet50")), {"learning_rate": 1e-3, "multi_gpu": False}),
    "dct_mlp": (DCTModel(model=MLP()), {"multi_gpu": False, "learning_rate": 1e-3, "weight_decay": 0.02}),
    "resnet18": (ResNet("resnet18"), {}),
    "resnet50": (ResNet("resnet50"), {}),
    "vgg11": (VGG("vgg11"), {}),
    "vgg11_bn": (VGG("vgg11_bn"), {}),
}


AUGMENTATIONS = {
    "no_aug": (NO_AUGMENT_TRANSFORM, NO_AUGMENT_TRANSFORM_DCT),
    "gaussian": (GAUSSIAN_BLUR_TRANSFORM, GAUSSIAN_BLUR_TRANSFORM_DCT),
    "jpeg": (JPEG_TRANSFORM, JPEG_TRANSFORM_DCT),
    "blur_jpeg_5": (blur_jpeg(.5), blur_jpeg(.5, dct=True)),
    "blur_then_jpeg_5": (blur_then_jpeg(.5), blur_then_jpeg(.5, dct=True)),
    "blur_jpeg_1": (blur_jpeg(.1), blur_jpeg(.1, dct=True)),
    "blur_then_jpeg_1": (blur_then_jpeg(.1), blur_then_jpeg(.1, dct=True)),
}


def _train(models, global_train_kwargs, augmentations, data_path, limit_data, limit_directories, experiment_name=None):
    for aug_name, aug in augmentations:
        logging.info("================================")
        logging.info(f"Training: {aug_name}")
        logging.info("================================")

        # load data with augment
        logging.info("Loading data...")

        for name, (model, train_kwargs) in models:
            logging.info("================================")
            logging.info(f"Starting Training {name}!")
            logging.info("================================")

            if "dct" in name:
                train_aug = aug[1]
            else:
                train_aug = aug[0]

            train, test = load_data_from_multiple(
                data_path, transformations=train_aug, limit_data=limit_data, limit_directories=limit_directories)

            logging.info(
                f"Loaded {len(train)} training images and {len(test)} testing images!")

            for key, value in global_train_kwargs.items():
                if key in train_kwargs:
                    continue
                train_kwargs[key] = value

            train_kwargs.update(
                {"save_path": f"experiments/{'' if experiment_name is None else experiment_name}/{limit_directories}_datasets/{aug_name}/{name}" if limit_data is None else "debug",
                 })
            with Training(model, train, **train_kwargs) as trainer:
                train_kwargs.update()
                trainer.train()
                acc = trainer.evaluate(test)
                logging.info(f"Final testing accuracy: {acc}")

            logging.info("================================")


def main(args):
    setup_logging()
    set_seed(42)

    save_dir = "experiments"

    global_train_kwargs = {
        "early_stopping_method": "patience",
        "multi_gpu": not args.no_multi_gpu,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "save_logits_histograms": args.save_histograms,
        "print_steps": 5 if args.debug else 100,
    }
    models = list(map(lambda x: (x, EVAL_MODELS[x]), args.models.split(",")))
    augmentations = list(
        map(lambda x: (x, AUGMENTATIONS[x]), args.augmentations.split(",")))

    limit_data = 200 if args.debug else None
    limit_directories = args.max_datasets
    _train(
        models=models,
        global_train_kwargs=global_train_kwargs,
        augmentations=augmentations,
        data_path=args.DATA,
        limit_data=limit_data,
        limit_directories=limit_directories,
        experiment_name=args.experiment_name,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=r"""
    Automatically train multiple models. Note that DCT models are trained using one GPU only since
    the MaxMin scaling layer only supports single GPU mode at the moment.
    """)

    parser.add_argument("DATA", help="Directory containing data.", type=str)
    parser.add_argument(
        "--patience", "-p",  help="Patience for early stopping.", default=5, type=int)
    parser.add_argument(
        "--batch_size", "-b", help="Batch size to use.", default=64, type=int)
    parser.add_argument(
        "--debug", "-d", help="Debug mode.", action="store_true")
    parser.add_argument(
        "--no_multi_gpu", "-g", help="Disable multi gpu training.", action="store_true")
    parser.add_argument(
        "--save_histograms", help="Save logits histograms.", action="store_true")
    parser.add_argument(
        "--max_datasets", help="Maximum datasets to use.", type=int, default=None)
    parser.add_argument(
        "--experiment_name", help="Provide name for experiment folder.", type=str, default=None)

    # =================================================0
    # Models
    # =================================================0
    models = ",".join(EVAL_MODELS.keys())
    default_model = "resnet50"
    parser.add_argument(
        "--models", "-m", help=f"""Models to train.
        You can either provide a single string (resnet50) or select multiple (resnet50,resnet18).
        Default: {default_model}
        Available: {models}
        """, default=default_model, type=str)

    # =================================================0
    # Augmentations
    # =================================================0
    augments = ",".join(AUGMENTATIONS.keys())
    default_augment = "blur_jpeg_5"
    parser.add_argument(
        "--augmentations", "-a", help=f"""Augmentations to use.
        You can either provide a single string (blur_jpeg) or select multiple (no_aug,gaussian).
        Default: {default_augment}
        Available: {augments}
        """, default=default_augment, type=str)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
