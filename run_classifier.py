import argparse

import torch

from gandetect.dataloader import ImageDataset
from gandetect.transforms import TEST_TRANSFORM, TEST_TRANSFORM_DCT
from gandetect.utils import set_seed


def main(args):
    set_seed(42)
    transform = TEST_TRANSFORM_DCT if args.dct else TEST_TRANSFORM

    data = ImageDataset(args.DIR, transformations=transform)
    model = torch.load(args.MODEL)
    model = model.eval()
    if args.cpu:
        model = model.cpu()

    for path, img in zip(data.paths, data):
        img = img.unsqueeze(0).to("cpu" if args.cpu else "cuda")
        fake = model(img).sigmoid().cpu() > .5
        print(f"{path} is {'fake' if fake else 'real'}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "MODEL", help="The model checkpoint to load.", type=str)
    parser.add_argument("DIR", help="The directory to evaluate.", type=str)
    parser.add_argument("--dct", help="Use a DCT model?.", action="store_true")
    parser.add_argument("--cpu", help="Run on CPU?.", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
