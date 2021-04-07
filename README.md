# [RE] CNN-generated images are surprisingly easy to spot...for now

<p>
<img align="right" width="250"  src="media/checkfant.png"> 
</p>

This work evaluates the reproducibility of the paper ``CNN-generated images are surprisingly easy to spot... for now'' by Wang et al. published at CVPR 2020.
The paper addresses the challenge of detecting CNN-generated imagery, which has reached the potential to even fool humans.
The authors propose two methods which help an image classifier to generalize from being trained on one specific CNN to detecting imagery produced by unseen architectures, training methods, or data sets.

## Setup

You can simply install all needed dependencies by running:

```
pip install -r requirements.txt
```


## Training models

Download instructions for the training set can found in the original [repository](https://github.com/peterwang512/CNNDetection).
You can also use our newly created [training set](https://drive.google.com/drive/folders/1zn0RbcxrhXfHMiR5EULxZWjwRAPVrEQk?usp=sharing).
Then you can utilize our training script:

```
python train_network.py -h

usage: train_network.py [-h] [--patience PATIENCE] [--batch_size BATCH_SIZE]
                        [--debug] [--no_multi_gpu] [--save_histograms]
                        [--max_datasets MAX_DATASETS]
                        [--experiment_name EXPERIMENT_NAME] [--models MODELS]
                        [--augmentations AUGMENTATIONS]
                        DATA

Automatically train multiple models. Note that DCT models are trained using
one GPU only since the MaxMin scaling layer only supports single GPU mode at
the moment.

positional arguments:
  DATA                  Directory containing data.

optional arguments:
  -h, --help            show this help message and exit
  --patience PATIENCE, -p PATIENCE
                        Patience for early stopping.
  --batch_size BATCH_SIZE, -b BATCH_SIZE
                        Batch size to use.
  --debug, -d           Debug mode.
  --no_multi_gpu, -g    Disable multi gpu training.
  --save_histograms     Save logits histograms.
  --max_datasets MAX_DATASETS
                        Maximum datasets to use.
  --experiment_name EXPERIMENT_NAME
                        Provide name for experiment folder.
  --models MODELS, -m MODELS
                        Models to train. You can either provide a single
                        string (resnet50) or select multiple
                        (resnet50,resnet18). Default: resnet50 Available: dct_
                        resnet18,dct_resnet50,dct_mlp,resnet18,resnet50,vgg11,
                        vgg11_bn
  --augmentations AUGMENTATIONS, -a AUGMENTATIONS
                        Augmentations to use. You can either provide a single
                        string (blur_jpeg) or select multiple
                        (no_aug,gaussian). Default: blur_jpeg_5 Available:
                        no_aug,gaussian,jpeg,blur_jpeg_5,blur_jpeg_1
```

**Example**

If you want to train a ResNet50 and VGG-11-BN model on the data in /data using all available augmentations and 8 classes:

`python train_network.py /data --augmentations no_aug,gaussian,jpeg,blur_jpeg_5,blur_jpeg_1 --models resnet50,vgg11_bn --max_datasets 8`



## Evaluaion

You can find our pre-trained models [online](https://drive.google.com/drive/folders/1zn0RbcxrhXfHMiR5EULxZWjwRAPVrEQk?usp=sharing).
For our evaluation we used the original [test data](https://github.com/peterwang512/CNNDetection).
Please rename the `whichfaceisreal` directory to `whichface`, since we assign labels based on the path.
The script autodetects all `checkpoint.pth` files in the path specified as the first argument.
It then evaluates them against the data stored in DIR.

```
python evaluate.py -h

usage: evaluate.py [-h] [--batch_size BATCH_SIZE] [--include_new] [--dct]
                   [--result_file_name RESULT_FILE_NAME]
                   MODELS DIR

positional arguments:
  MODELS                Model to use.
  DIR                   Directory to analyse.

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        Batch size to use default: 64.
  --include_new         Includes StyleGAN2 and Whichfaceisreal in the
                        evaluation.
  --dct                 Use DCT preprocessing instead.
  --result_file_name RESULT_FILE_NAME, -r RESULT_FILE_NAME
                        Result file name for csv/tex; default: {results_file}.
```

**Example**

If you want to evaluate all model stored in ./experiments on the data in /data/test:

`python evaluate.py ./experiments  /data/test`
