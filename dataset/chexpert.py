'''
This script creates a dataset and configures the dataloader according to desired
data transformation inputs
'''

import pandas as pd
import os
import sys
import torch
import cv2
import util
sys.path.append(os.getcwd())

from constants        import *
from torch.utils.data import Dataset, DataLoader
from torchvision      import transforms
from typing           import Union
from pathlib          import Path
from PIL              import Image


class CheXpertDataset(Dataset):
    """Template dataset class
    Args:
        data_path (str, Path): path to dataset
    """

    def __init__(
            self,
            csv_path: Union[str, Path],
            data_transform: transforms.Compose,
            img_type: str = "all", 
            uncertain: str = "ignore",
            split: str = "train"
        ):
        """Constructor for dataset class

        Args: 
            data_path (str): path to csv file containing paths to jpgs
            data_transfrom (transforms.Compose): data transformations
            img_type (str): type of xray images to use ["All", "Frontal", "Lateral"]
            uncertain (str): how to handel uncertain cases ["ignore", "zero", "one"]
            split (str): datasplit to parse in a dataloader
        """
        # read in csv file
        self.df = pd.read_csv(csv_path)

        # filter image type 
        if img_type != "All":
            self.df = self.df[self.df['Frontal/Lateral'] == img_type]

        # get column names of the target labels
        self.label_cols = self.df.columns[-14:]

        # fill na with 0s
        self.df = self.df.fillna(0)

        #this changes the uncertain labels according to the parameter uncertain
        if uncertain == "ignore": 
            self.df["Remove"] = self.df.apply(lambda x: 1 if -1 in list(x[self.label_cols]) else 0, axis=1)
            self.df = self.df[~(self.df.Remove == 1)]
        elif uncertain == "zero": 
            self.df['Labels'].loc[self.df['Labels'] == -1.] = 0.0
        elif uncertain == "one": 
            self.df['Labels'].loc[self.df['Labels'] == -1.] = 0.0

        self.data_transform = data_transform

    def __len__(self):
        '''Returns the size of the dataset'''
        return len(self.df)

    def __getitem__(self, idx):
        '''
        Params:
            idx (integer): the index of the image we want to look at
        Returns:
            x (array): the transformed numpy array representing the image we want
            y (list): list of labels associated with the image
        '''

        # get prediction labels
        y = list(self.df.iloc[idx][list(self.label_cols)])
        y = torch.tensor(y)

        # get images
        path = CHEXPERT_DATA_DIR / self.df.iloc[idx]["Path"]
        x = cv2.imread(str(path), 0)

        # tranform images 
        x = util.resize_img(x, 256)
        x = Image.fromarray(x).convert('RGB')
        if (self.data_transform is not None) and self.split == "train":
            x = self.data_transform(x)

        return x, y


def get_dataloader(args, split):
    '''Defines augmentations
    Initializes Dataset with augmentations & Passes in dataset to Dataloader

    Params: 
        args (argsparse.ArgumentParser): arguments
        split (str): dataset split for the dataloader
    Returns:
        dataloader (dataloader): dataloader configured with desired data augmentations
    '''

    # transformations 
    # TODO: maybe this deserves its own function
    data_transform = transforms.Compose([
        transforms.Resize((args.resize_shape, args.resize_shape)),
        transforms.RandomCrop((args.crop_shape, args.crop_shape)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD)
    ])

    # initialize dataset class
    csv_path = CHEXPERT_DATA_DIR / f"{split}.csv"
    dataset = CheXpertDataset(
        csv_path = csv_path, 
        img_type = args.img_type, 
        data_transform = data_transform,
        uncertain = args.uncertain,
        split = split
    )
    
    # create dataloader
    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )

    return dataloader
