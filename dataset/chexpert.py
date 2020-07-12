'''
This script creates a dataset and configures the dataloader according to desired
data transformation inputs
'''

import numpy as np
import pandas as pd
import pickle
import os
import sys
import torch
import cv2
import util
import torchvision.transforms as t
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

    def __init__(self,
                 data_path: Union[str, Path],
                 data_transform = None,
                 img_type:str = "All", 
                 uncertain: str = "ignore"):
        ''' Constructor for dataset class
        Args: 
            data_path(csv file): path to csv file containing paths to jpgs
        '''
        # read in csv file
        self.df = pd.read_csv(data_path)

        # filter image type 
        if img_type != "All":
            self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']

        self.data_path = data_path

        # get column names of the target labels
        self.label_cols = self.df.columns[-14:]

        # fill na with 0s
        self.df = self.df.fillna(0)

        #this changes the uncertain labels according to the parameter uncertain
        if uncertain == "ignore": 
            self.df["Remove"] = self.df.apply(lambda x: 1 if -1 in list(x[self.label_cols]) else 0, axis=1)
            self.df = self.df[~(self.df.Remove == 1)]
        elif uncertain == "zero": 
            # TODO
            pass
        elif uncertain == "one": 
            # TODO
            pass

        self.data_transform = data_transform


    def __len__(self):
        '''Returns the size of the dataset
        '''
        return len(self.df)


    def __getitem__(self, idx):
        '''
        Params:
            idx (integer): the index of the image we want to look at
        Returns:
            x (array): the transformed numpy array representing the image we want
            y (list): list of labels associated with the image
        '''

        y = list(self.df.iloc[idx][list(self.label_cols)])

        path = CHEXPERT_DATA_DIR / self.df.iloc[idx]["Path"]
        x = cv2.imread(str(path), 0)
        if x is None: 
            print(path)
        x = util.resize_img(x, 256)
        x = Image.fromarray(x).convert('RGB')

        if self.data_transform is not None:
            x = self.data_transform(x)

        y = torch.tensor(y)

        return x, y


    def resize_img(self, img, scale):
        """TODO: 
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)
        if max_ind == 0:
            # width fixed at scale
            wpercent = (scale / float(size[0]))
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # height fixed at scale
            hpercent = (scale / float(size[1]))
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)

        resized_img = cv2.resize(img, desireable_size[::-1])

        return resized_img


def get_dataloader(dataloader_args, dataset_args):
    '''Defines augmentations
    Initializes Dataset class with augmentations 
    Passes in dataset to Dataloader
    Params: 
        dataloader_args (dict): input parameters for the dataloader
        dataset_args (dict): input parameters for the dataset
    Returns:
        dataloader (dataloader): dataloader configured with desired data augmentations
    '''

    # initialize dataset class
    dataset = CheXpertDataset(**dataset_args)
    
    # create dataloader
    dataloader = DataLoader(dataset, **dataloader_args)
    return dataloader
