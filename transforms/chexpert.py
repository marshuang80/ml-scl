from warnings import warn

import PIL
import numpy as np

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