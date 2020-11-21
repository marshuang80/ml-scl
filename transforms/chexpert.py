"""
Self-supervision transformations for CheXpert 

Adapted from: 
    github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/models/self_supervised/
"""

from warnings import warn

import PIL
import numpy as np
from typing import List


try:
    import torchvision.transforms as transforms
except ModuleNotFoundError:
    warn('You want to use `torchvision` which is not installed yet,'  # pragma: no-cover
         ' install it with `pip install torchvision`.')
    _TORCHVISION_AVAILABLE = False
else:
    _TORCHVISION_AVAILABLE = True

try:
    import cv2
except ModuleNotFoundError:
    warn('You want to use `opencv-python` which is not installed yet,'  # pragma: no-cover
         ' install it with `pip install opencv-python`.')


class SimCLRTrainDataTransform(object):
    """
    Transforms for SimCLR
    Transform::
        RandomResizedCrop(size=self.input_height)
        RandomHorizontalFlip()
        RandomRotation()
        GaussianBlur(kernel_size=int(0.1 * self.input_height))
        transforms.ToTensor()
    """
    def __init__(self, input_height):
        if not _TORCHVISION_AVAILABLE:
            raise ModuleNotFoundError(  # pragma: no-cover
                'You want to use `transforms` from `torchvision` which is not installed yet.'
            )

        self.input_height = input_height
        data_transforms = transforms.Compose([transforms.RandomCrop(size=self.input_height),
                                              transforms.RandomRotation(20, resample=PIL.Image.BICUBIC),
                                              transforms.RandomHorizontalFlip(),
                                              GaussianBlur(kernel_size=int(0.1 * 256)),   # TODO
                                              transforms.ToTensor()])
        self.train_transform = data_transforms

    def __call__(self, sample):
        transform = self.train_transform
        xi = transform(sample)
        xj = transform(sample)
        return xi, xj


class SimCLREvalDataTransform(object):
    """
    Transforms for SimCLR
    Transform::
        transforms.CenterCrop(input_height),
        transforms.ToTensor()
    """
    def __init__(self, input_height):
        if not _TORCHVISION_AVAILABLE:
            raise ModuleNotFoundError(  # pragma: no-cover
                'You want to use `transforms` from `torchvision` which is not installed yet.'
            )

        self.input_height = input_height
        self.test_transform = transforms.Compose([
            transforms.CenterCrop(input_height),
            transforms.ToTensor(),
        ])

    def __call__(self, sample):
        transform = self.test_transform
        xi = transform(sample)
        xj = transform(sample)
        return xi, xj


class SwAVTrainDataTransform(object):
    def __init__(
        self,
        size_crops: List[int] = [224, 96],
        nmb_crops: List[int] = [2, 4],
        min_scale_crops: List[float] = [0.33, 0.10],
        max_scale_crops: List[float] = [1, 0.33],
        gaussian_blur: bool = True,
    ):
        self.gaussian_blur = gaussian_blur

        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)

        self.size_crops = size_crops
        self.nmb_crops = nmb_crops
        self.min_scale_crops = min_scale_crops
        self.max_scale_crops = max_scale_crops

        transform = []
        img_transform = [
            transforms.RandomCrop(size=self.input_height),
            transforms.RandomRotation(20, resample=PIL.Image.BICUBIC),
        ]

        if self.gaussian_blur:
            kernel_size = int(0.1 * self.size_crops[0])
            if kernel_size % 2 == 0:
                kernel_size += 1

            img_transform.append(
                GaussianBlur(kernel_size=kernel_size, p=0.5)
            )

        self.img_transform = transforms.Compose(img_transform)
        self.final_transform = transforms.Compose([transforms.ToTensor()])


        for i in range(len(self.size_crops)):
            random_resized_crop = transforms.RandomResizedCrop(
                self.size_crops[i],
                scale=(self.min_scale_crops[i], self.max_scale_crops[i]),
            )

            transform.extend([transforms.Compose([
                random_resized_crop,
                transforms.RandomHorizontalFlip(p=0.5),
                self.img_transform,
                self.final_transform])
            ] * self.nmb_crops[i])

        self.transform = transform

        # add online train transform of the size of global view
        online_train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.size_crops[0]),
            transforms.RandomHorizontalFlip(),
            self.final_transform
        ])

        self.transform.append(online_train_transform)

    def __call__(self, sample):
        multi_crops = list(
            map(lambda transform: transform(sample), self.transform)
        )

        return multi_crops


class SwAVEvalDataTransform(SwAVTrainDataTransform):
    def __init__(
        self,
        size_crops: List[int] = [224, 96],
        nmb_crops: List[int] = [2, 4],
        min_scale_crops: List[float] = [0.33, 0.10],
        max_scale_crops: List[float] = [1, 0.33],
        gaussian_blur: bool = True,
    ):
        super().__init__(
            size_crops=size_crops,
            nmb_crops=nmb_crops,
            min_scale_crops=min_scale_crops,
            max_scale_crops=max_scale_crops,
            gaussian_blur=gaussian_blur,
        )

        input_height = self.size_crops[0]  # get global view crop
        test_transform = transforms.Compose([
            transforms.Resize(int(input_height + 0.1 * input_height)),
            transforms.CenterCrop(input_height),
            self.final_transform,
        ])

        # replace last transform to eval transform in self.transform list
        self.transform[-1] = test_transform



class GaussianBlur(object):
    """
    Implements Gaussian blur as described in the SimCLR paper
    """

    def __init__(self, kernel_size, min=0.1, max=2.0):
        if not _TORCHVISION_AVAILABLE:
            raise ModuleNotFoundError(  # pragma: no-cover
                'You want to use `transforms` from `torchvision` which is not installed yet.'
            )

        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample
