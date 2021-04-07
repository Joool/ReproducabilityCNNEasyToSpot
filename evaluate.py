import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from gandetect.dataloader import load_data
from gandetect.transforms import TEST_TRANSFORM, TEST_TRANSFORM_DCT
from gandetect.utils import set_seed

EVAL_DATASETS = {
    "ProGAN": "progan",
    "StyleGAN": "stylegan",
    "StyleGAN2": "stylegan2",
    "BigGAN": "biggan",
    "CycleGAN": "cyclegan",
    "StarGAN": "stargan",
    "GauGAN": "gaugan",
    "CRN": "crn",
    "IMLE": "imle",
    "SITD": "seeingdark",
    "SAN": "san",
    "DeepFake": "deepfake",
    "Whichfaceisreal": "whichface",
}


def main(args):
    set_seed(42)
    models_paths = [str(path.absolute())
                    for path in Path(args.MODELS).rglob("checkpoint.pth")]

    # prep datasets
    datasets = {name: f"{args.DIR}/{path}"
                for name, path in EVAL_DATASETS.items()}

    if not args.include_new:
        del datasets["StyleGAN2"]
        del datasets["Whichfaceisreal"]

    transfrom = TEST_TRANSFORM_DCT if args.dct else TEST_TRANSFORM
    datasets = {name: torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(
            load_data(dataset_path, transformations=transfrom)),
        num_workers=mp.cpu_count(), batch_size=args.batch_size) for name, dataset_path in datasets.items()}

    with open(f"{args.result_file_name}.csv", "w+") as csv, open(f"{args.result_file_name}.tex", "w+") as tex:
        # headers
        csv.write(f"Name,{','.join(datasets.keys())},mAP\n")

        tex.write(r"\begin{tabular}{" + f"{'cc' + len(datasets)* 'c'}"+"}\n")
        tex.write("\\toprule\n")
        tex.write(r"Name &")
        for name in datasets.keys():
            tex.write(f"{name} &")
        tex.write(r"mAP \\" + "\n")

        for model_path in models_paths:
            print(f"Evaluating {model_path}")
            model = torch.load(model_path)
            model = model.eval()

            csv.write(f"{model_path},")

            # name needs to be changed before table compiles
            tex.write(f"{model_path} &")

            device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")

            mAP = []
            print(f"============================")
            with torch.no_grad():
                for name, dataloader in datasets.items():
                    print(f"Evaluating {name}!")
                    y_true, y_pred = [], []
                    for batch in tqdm(dataloader):
                        imgs, labels = batch
                        imgs = imgs.to(device)
                        predictions = model(
                            imgs).sigmoid().float().flatten().tolist()

                        y_true.extend(labels.float().flatten().tolist())
                        y_pred.extend(predictions)

                    ap = average_precision_score(y_true=y_true, y_score=y_pred)
                    print(f"{name} - {ap:.2%}")
                    csv.write(f"{ap * 100:.1f},")
                    tex.write(f"{ap * 100:.1f} &")
                    mAP.append(ap)
                    print(f"============================")

            mAP = np.mean(mAP)
            print(f"mAP: {mAP:.2%}")
            csv.write(f"{mAP * 100:.1f}\n")
            tex.write(f"{mAP * 100:.1f}" + r"\\ \midrule" + "\n")

        tex.write(r"\end{tabular}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("MODELS", help="Model to use.", type=str)
    parser.add_argument("DIR", help="Directory to analyse.", type=str)

    default_batch = 64
    parser.add_argument(
        "--batch_size", help=f"Batch size to use default: {default_batch}.", default=default_batch)

    parser.add_argument(
        "--include_new", help="Includes StyleGAN2 and Whichfaceisreal in the evaluation.", action="store_true")
    parser.add_argument(
        "--dct", help="Use DCT preprocessing instead.", action="store_true")

    results_file = "results"
    parser.add_argument(
        "--result_file_name", "-r", help="Result file name for csv/tex; default: {results_file}.", type=str, default=results_file)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
