"""
Image transformation module for data augmentation.

This module provides custom image transformation classes and utilities for data augmentation
in computer vision tasks. It includes a customized RandAugment implementation with selective
augmentation operations, a dynamic size transformation class, and a comprehensive transform
creation function.

The module is designed to work with PyTorch and torchvision, providing flexible and
configurable image augmentation pipelines suitable for training deep learning models.
"""

from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torchvision.transforms import RandomResizedCrop, RandAugment, Compose, ColorJitter, RandomHorizontalFlip, \
    RandomVerticalFlip, RandomApply, RandomChoice, RandomRotation


class MyRandAugment(RandAugment):
    """
    Custom RandAugment implementation with selective augmentation operations.

    This class extends torchvision's RandAugment to provide a curated set of
    augmentation operations that are commonly effective for image classification
    tasks while excluding potentially problematic transformations.

    The augmentation space includes color-based transformations (brightness, color,
    contrast, sharpness), histogram operations (posterize, autocontrast, equalize),
    and identity operation, while excluding geometric transformations that might
    be handled separately.

    Example::
        >>> transform = MyRandAugment(num_ops=2, magnitude=9)
        >>> augmented_image = transform(image)
    """

    def _augmentation_space(self, num_bins: int, image_size: tuple[int, int]) -> dict[str, tuple[Tensor, bool]]:
        """
        Define the augmentation space with available operations and their parameters.

        :param num_bins: Number of magnitude bins for each operation.
        :type num_bins: int
        :param image_size: Size of the input image as (height, width).
        :type image_size: tuple[int, int]

        :return: Dictionary mapping operation names to (magnitudes, signed) tuples.
        :rtype: dict[str, tuple[Tensor, bool]]
        """
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            # "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            # "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            # "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
            # "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
            # "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            # "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
            # "Hue": (torch.linspace(0.0, 0.5, num_bins), True),
        }


class SizeTrans:
    """
    Dynamic size transformation class for images with stochastic resizing.

    This class provides a flexible approach to image resizing by sampling target
    dimensions from normal distributions with configurable parameters. It combines
    random size selection with aspect ratio variation to create diverse training
    samples while maintaining control over the output dimensions.

    The transformation works by:
    1. Sampling a target size from a normal distribution
    2. Sampling an aspect ratio from another normal distribution
    3. Computing width and height based on the sampled values
    4. Applying RandomResizedCrop with the computed dimensions

    Example::
        >>> size_trans = SizeTrans(size_mean=224, size_std=50, ratio_mean=1.0, ratio_std=0.2)
        >>> transformed_image = size_trans(image)
    """

    def __init__(self, size_mean: int = 360, size_std: int = 90,
                 size_min: Optional[int] = 66, size_max: Optional[int] = None,
                 ratio_mean: float = 1.0, ratio_std: float = 0.25,
                 ratio_min: Optional[float] = 0.25, ratio_max: Optional[float] = 4.0,
                 scale: Optional[Tuple[float, float]] = (0.5, 1.0),
                 ratio: Optional[Tuple[float, float]] = (0.8, 1.2)):
        """
        Initialize the SizeTrans transformation.

        :param size_mean: Mean value for the size distribution.
        :type size_mean: int
        :param size_std: Standard deviation for the size distribution.
        :type size_std: int
        :param size_min: Minimum allowed size (clipping lower bound).
        :type size_min: Optional[int]
        :param size_max: Maximum allowed size (clipping upper bound).
        :type size_max: Optional[int]
        :param ratio_mean: Mean value for the aspect ratio distribution.
        :type ratio_mean: float
        :param ratio_std: Standard deviation for the aspect ratio distribution.
        :type ratio_std: float
        :param ratio_min: Minimum allowed aspect ratio (clipping lower bound).
        :type ratio_min: Optional[float]
        :param ratio_max: Maximum allowed aspect ratio (clipping upper bound).
        :type ratio_max: Optional[float]
        :param scale: Scale range for RandomResizedCrop.
        :type scale: Optional[Tuple[float, float]]
        :param ratio: Aspect ratio range for RandomResizedCrop.
        :type ratio: Optional[Tuple[float, float]]
        """
        self.size_mean = size_mean
        self.size_std = size_std
        self.size_min = size_min
        self.size_max = size_max

        self.ratio_mean = ratio_mean
        self.ratio_std = ratio_std
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max

        self.scale = scale
        self.ratio = ratio

    def _get_trans(self):
        """
        Generate a RandomResizedCrop transform with stochastically sampled dimensions.

        :return: A RandomResizedCrop transform with computed target size.
        :rtype: RandomResizedCrop
        """
        s = np.random.normal(self.size_mean, self.size_std)
        if self.size_min is not None:
            s = max(s, self.size_min)
        if self.size_max is not None:
            s = min(s, self.size_max)

        r = np.random.normal(self.ratio_mean, self.ratio_std)
        if self.ratio_min is not None:
            r = max(r, self.ratio_min)
        if self.ratio_max is not None:
            r = min(r, self.ratio_max)

        w, h = int(s * (r ** 0.5)), int(s / (r ** 0.5))
        return RandomResizedCrop(
            size=(w, h),
            scale=self.scale,
            ratio=self.ratio,
        )

    def __call__(self, image):
        """
        Apply the transformation to an image.

        :param image: Input image to be transformed.
        :type image: PIL.Image or torch.Tensor

        :return: Transformed image.
        :rtype: PIL.Image or torch.Tensor
        """
        return self._get_trans()(image)


def create_transform(
        hue: float = 0.2,
        num_ops: int = 4,
        magnitude: int = 8,
        horizontal_flip_p: float = 0.5,
        vertical_flip_p: float = 0.5,
        rotation_p: float = 0.3,
        size_mean: int = 360,
        size_std: int = 90,
        size_min: Optional[int] = 66,
        size_max: Optional[int] = None,
        ratio_mean: float = 1.0,
        ratio_std: float = 0.25,
        ratio_min: Optional[float] = 0.25,
        ratio_max: Optional[float] = 4.0,
        scale: Optional[Tuple[float, float]] = (0.5, 1.0),
        ratio: Optional[Tuple[float, float]] = (0.8, 1.2)
):
    """
    Create a comprehensive image transformation pipeline for data augmentation.

    This function constructs a sequential composition of image transformations
    designed for robust data augmentation in computer vision tasks. The pipeline
    includes color adjustments, random augmentation operations, geometric
    transformations, and dynamic resizing.

    The transformation sequence:
    1. ColorJitter for hue adjustment
    2. MyRandAugment for various augmentation operations
    3. Random horizontal and vertical flips
    4. Random rotation (90°, 180°, or 270°) with configurable probability
    5. Dynamic size transformation with stochastic dimensions

    :param hue: ColorJitter hue parameter for color variation.
    :type hue: float
    :param num_ops: Number of operations for MyRandAugment.
    :type num_ops: int
    :param magnitude: Magnitude parameter for MyRandAugment operations.
    :type magnitude: int
    :param horizontal_flip_p: Probability of horizontal flip.
    :type horizontal_flip_p: float
    :param vertical_flip_p: Probability of vertical flip.
    :type vertical_flip_p: float
    :param rotation_p: Probability of applying rotation.
    :type rotation_p: float
    :param size_mean: Mean value for SizeTrans size distribution.
    :type size_mean: int
    :param size_std: Standard deviation for SizeTrans size distribution.
    :type size_std: int
    :param size_min: Minimum size for SizeTrans.
    :type size_min: Optional[int]
    :param size_max: Maximum size for SizeTrans.
    :type size_max: Optional[int]
    :param ratio_mean: Mean aspect ratio for SizeTrans.
    :type ratio_mean: float
    :param ratio_std: Standard deviation for aspect ratio in SizeTrans.
    :type ratio_std: float
    :param ratio_min: Minimum aspect ratio for SizeTrans.
    :type ratio_min: float
    :param ratio_max: Maximum aspect ratio for SizeTrans.
    :type ratio_max: float
    :param scale: Scale range for SizeTrans RandomResizedCrop.
    :type scale: Tuple[float, float]
    :param ratio: Aspect ratio range for SizeTrans RandomResizedCrop.
    :type ratio: Tuple[float, float]

    :return: Composed transformation pipeline.
    :rtype: Compose

    Example::
        >>> transform = create_transform(hue=0.1, num_ops=2, magnitude=5)
        >>> augmented_image = transform(original_image)
    """
    return Compose([
        ColorJitter(
            brightness=0.0,  # Fixed value
            contrast=0.0,  # Fixed value
            saturation=0.0,  # Fixed value
            hue=hue,
        ),
        MyRandAugment(num_ops=num_ops, magnitude=magnitude),
        RandomHorizontalFlip(p=horizontal_flip_p),
        RandomVerticalFlip(p=vertical_flip_p),
        RandomApply([
            RandomChoice([
                RandomRotation(degrees=(90, 90), expand=True),  # Fixed value
                RandomRotation(degrees=(180, 180), expand=True),  # Fixed value
                RandomRotation(degrees=(270, 270), expand=True),  # Fixed value
            ])
        ], p=rotation_p),
        SizeTrans(
            size_mean=size_mean,
            size_std=size_std,
            size_min=size_min,
            size_max=size_max,
            ratio_mean=ratio_mean,
            ratio_std=ratio_std,
            ratio_min=ratio_min,
            ratio_max=ratio_max,
            scale=scale,
            ratio=ratio,
        ),
    ])


def create_transform_with_fixed_size(
        hue: float = 0.2,
        num_ops: int = 4,
        magnitude: int = 8,
        horizontal_flip_p: float = 0.5,
        vertical_flip_p: float = 0.5,
        rotation_p: float = 0.3,
        size: Optional[Tuple[int, int]] = (640, 640),
        scale: Optional[Tuple[float, float]] = (0.5, 1.0),
        ratio: Optional[Tuple[float, float]] = (0.8, 1.2)
):
    return Compose([
        ColorJitter(
            brightness=0.0,  # Fixed value
            contrast=0.0,  # Fixed value
            saturation=0.0,  # Fixed value
            hue=hue,
        ),
        MyRandAugment(num_ops=num_ops, magnitude=magnitude),
        RandomHorizontalFlip(p=horizontal_flip_p),
        RandomVerticalFlip(p=vertical_flip_p),
        RandomApply([
            RandomChoice([
                RandomRotation(degrees=(90, 90), expand=True),  # Fixed value
                RandomRotation(degrees=(180, 180), expand=True),  # Fixed value
                RandomRotation(degrees=(270, 270), expand=True),  # Fixed value
            ])
        ], p=rotation_p),
        RandomResizedCrop(
            size=size,
            scale=scale,
            ratio=ratio,
        ),
    ])
