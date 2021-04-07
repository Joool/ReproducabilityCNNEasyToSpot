import argparse

import torch
from PIL import Image

from gandetect.dataloader import image_paths
from gandetect.transforms import TEST_TRANSFORM
from gandetect.utils import set_seed


def main(args):
    set_seed(42)
    model = torch.load(args.MODEL)
    model = model.eval()

    images = image_paths(args.DIR)

    transformed = map(Image.open, images)
    transformed = map(TEST_TRANSFORM, transformed)

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    with torch.no_grad():
        for i, img in enumerate(transformed):
            img = img.unsqueeze(0).float().to(device)
            fake = torch.sigmoid(model(img)) > .5

            if fake:
                print(f"Image {images[i]} is generated!")
            else:

                print(f"Image {images[i]} is real!")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("MODEL", help="Model to use.", type=str)
    parser.add_argument("DIR", help="Directory to analyse.", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
