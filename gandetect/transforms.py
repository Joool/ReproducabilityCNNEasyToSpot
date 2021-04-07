from io import BytesIO

import cv2
import numpy as np
import torch
import torch_dct as dct
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.fft import dct
from torchvision import transforms

IMAGE_SIZE = 224
RESIZE_SIZE = 256


# =====================================================
# Helper functions
# =====================================================
_RNG = np.random.default_rng(42)


def _dct_transform_and_log_scale():
    def _dct_transform(x):
        # check channel first
        assert x.shape[0] == 3

        # compute dct over row and columns
        x = dct(x, type=2, norm="ortho", axis=-1)
        x = dct(x, type=2, norm="ortho", axis=-2)

        return x

    def _log_scale(x):
        x = np.abs(x)
        x += 1e-12  # no zero in log
        x = np.log(x)
        return x

    return [
        transforms.Lambda(_dct_transform),
        transforms.Lambda(_log_scale),
    ]


def _build_transform(data_augmentations, resize=False, training=False, dct=False, crop=True):
    # build transformations
    trans = list()

    # want to resize before cropping?
    if resize:
        trans.append(transforms.Resize(RESIZE_SIZE, Image.BICUBIC))

    # add data_augmentations
    trans += data_augmentations

    # training transformations?
    if training:
        trans += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(IMAGE_SIZE),
        ]
    elif crop:
        trans += [
            transforms.CenterCrop(IMAGE_SIZE),
        ]

    # DCT-converted input?
    if dct:
        # we do tranform or normalize dct data
        # the transformation and normalization is done
        # on-the-fly during training
        def _pil_to_tensor(pic):
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
            img = img.permute((2, 0, 1)).contiguous()
            return img.float()

        trans += [
            transforms.Lambda(_pil_to_tensor)
        ]
    else:
        trans += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]

    return transforms.Compose(
        trans
    )


def _jpeg(img: Image):
    # convention for JPEG image quality standard is [0,...,95]
    # However, the paper specifies sampling in [30, 100]
    quality = int(min(95, _RNG.integers(30, 101)))
    if _RNG.random() > .5:
        # cv2
        img_cv2 = np.asarray(img)[:, :, ::-1]
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
        img = cv2.imdecode(encimg, 1)
        img = img[:, :, ::-1]
        img = Image.fromarray(img)
    else:
        # PIL
        with BytesIO() as buf:
            img.save(buf, "JPEG", quality=quality)

            # load from buffer to memory
            img = Image.fromarray(np.array(Image.open(buf).convert("RGB")))

    return img


# =====================================================
# Transformations internal
# =====================================================
_GAUSSIAN_SIGMA = (1e-12, 3.)

_GAUSSIAN_BLUR_TRANSFORM = [
    transforms.RandomApply(torch.nn.ModuleList([
        transforms.GaussianBlur(kernel_size=3, sigma=_GAUSSIAN_SIGMA),
    ]), p=0.5)]


_JPEG_TRANSFORM = [
    transforms.Lambda(lambda x: x if _RNG.random() > .5 else _jpeg(x)),
]


def _blur_jpeg(p):
    return [transforms.Lambda(lambda x: x if _RNG.random() > p else _jpeg(
            TF.gaussian_blur(x, kernel_size=3, sigma=_GAUSSIAN_SIGMA)))]


def _blur_then_jpeg(p):
    return [transforms.Lambda(lambda x: x if _RNG.random() > p else
                              TF.gaussian_blur(x, kernel_size=3, sigma=_GAUSSIAN_SIGMA)),
            transforms.Lambda(lambda x: x if _RNG.random() > p else _jpeg(x))]


# =====================================================
# Transformations
# =====================================================
def blur_then_jpeg(p, dct=False):
    """Create a tranformation pipeline with the given p value.
    """
    return _build_transform(_blur_then_jpeg(p), training=True, dct=dct)


def blur_jpeg(p, dct=False):
    """Create a tranformation pipeline with the given p value.
    """
    return _build_transform(_blur_jpeg(p), training=True, dct=dct)


GAUSSIAN_BLUR_TRANSFORM = _build_transform(
    _GAUSSIAN_BLUR_TRANSFORM, training=True)

GAUSSIAN_BLUR_TRANSFORM_DCT = _build_transform(
    _GAUSSIAN_BLUR_TRANSFORM, training=True, dct=True)

JPEG_TRANSFORM = _build_transform(_JPEG_TRANSFORM, training=True)

JPEG_TRANSFORM_DCT = _build_transform(_JPEG_TRANSFORM, training=True, dct=True)

TEST_TRANSFORM = _build_transform([])

TEST_TRANSFORM_DCT = _build_transform([], dct=True)

NO_AUGMENT_TRANSFORM = _build_transform([], training=True)

NO_AUGMENT_TRANSFORM_DCT = _build_transform([], training=True, dct=True)
