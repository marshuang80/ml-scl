import os
import torchvision.transforms as transform_lib

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from typing import Optional 
from dataset.chexpert import CheXpertDataset
from constants   import *


class CheXpertDataModule(LightningDataModule):
    """https://stanfordmlgroup.github.io/competitions/chexpert/"""

    name = 'chexpert'

    def __init__(
            self,
            data_dir: str = '/data4/selfsupervision/chexpert/CheXpert/CheXpert-v1.0/', # TODO: remove default dir
            image_type: str = 'all',
            uncertain: str = 'ignore', 
            image_size: int = 256,
            crop_size: int = 224,
            num_workers: int = 16,
            batch_size: int = 32,
            *args,
            **kwargs,
    ):
        """
        Args:
            data_dir: path to the imagenet dataset file
            img_type (str): type of xray images to use ["All", "Frontal", "Lateral"]
            uncertain (str): how to handel uncertain cases ["ignore", "zero", "one"]
            img_size (int): final image size
            crop_size (int): crop size for imge 
            num_workers: how many data workers
            batch_size: batch_size

        """
        super().__init__(*args, **kwargs)

        self.image_size = image_size
        self.crop_size = crop_size
        self.dims = (3, self.image_size, self.image_size)
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.img_type = image_type
        self.uncertain = uncertain

        self.train_transforms = None
        self.val_transforms = None

    @property
    def num_classes(self):
        """CheXpert has 14 conditions"""
        return 14

    def _verify_splits(self, data_dir, split):
        dirs = os.listdir(data_dir)

        if split not in dirs:
            raise FileNotFoundError(f'a {split} Imagenet split was not found in {data_dir},'
                                    f' make sure the folder contains a subfolder named {split}')

    def prepare_data(self):
        """
        This method already assumes you have CheXpert downloaded.
        """
        self._verify_splits(self.data_dir, 'train')
        self._verify_splits(self.data_dir, 'valid')

    def train_dataloader(self):
        """Uses the train split of CheXpert"""
        transforms = self.train_transform() if self.train_transforms is None else self.train_transforms

        dataset = CheXpertDataset(
            csv_path=os.path.join(self.data_dir, 'train.csv'), 
            data_transform=transforms,
            img_type=self.img_type,
            uncertain=self.uncertain, 
            resize_shape=self.image_size
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def val_dataloader(self):
        """Uses the valid split of CheXpert"""
        transforms = self.val_transform() if self.val_transforms is None else self.val_transforms

        dataset = CheXpertDataset(
            csv_path=os.path.join(self.data_dir, 'valid.csv'), 
            data_transform=transforms,
            img_type=self.img_type,
            uncertain=self.uncertain, 
            resize_shape=self.image_size
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return loader

    def test_dataloader(self):
        """Uses the valid split of CheXpert"""
        transforms = self.val_transform() if self.val_transforms is None else self.val_transforms

        dataset = CheXpertDataset(
            csv_path=os.path.join(self.data_dir, 'test.csv'), 
            data_transform=transforms,
            img_type=self.img_type,
            uncertain=self.uncertain, 
            resize_shape=self.image_size
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return loader

    def train_transform(self):
        """The standard imagenet transforms"""
        preprocessing = transform_lib.Compose([
            transform_lib.RandomResizedCrop(self.crop_size),
            transform_lib.RandomHorizontalFlip(),
            transform_lib.ToTensor(),
            transform_lib.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        return preprocessing

    def val_transform(self):
        """The standard imagenet transforms for validation"""

        preprocessing = transform_lib.Compose([
            transform_lib.CenterCrop(self.image_size),
            transform_lib.ToTensor(),
            transform_lib.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        return preprocessing