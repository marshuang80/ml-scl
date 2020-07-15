import PIL
from torch import imag
import torchvision.transforms as t
import cv2
import torch
import numpy as np
from PIL         import Image, ImageFilter, ImageOps
from constants   import *


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class GaussianNoise:
    '''Add gaussian noise according to the mean and std dev'''

    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std = std
        
    def __call__(self, img):

        std_unit = self.std / 5
        list_std = np.array([std_unit * factor for factor in range(6)])
        list_prob = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
        chosen_weight = np.random.choice(list_std, p=list_prob)
        
        img = img + torch.randn(img.size()) * chosen_weight + self.mean
        return img
    
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'


class GaussianBlur(object):
    '''Add gaussian blur according to the radius'''

    def __init__(self, radius=1.0):
        self.radius = radius
    
    def __call__(self, img):
        '''
        Splits given radius into 5 increments and choses a radius increment with weighted probability
        Applies a gaussian blur with chosen radius to PILimage and returns the blurred image 
        '''
        radius_unit = self.radius / 5 
        list_radius = np.array([radius_unit * factor for factor in range(6)])
        list_prob = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
        chosen_radius = np.random.choice(list_radius, p=list_prob)
        img = img.filter(ImageFilter.GaussianBlur(radius = chosen_radius))
        return img
    
    def __repr__(self):
        return self.__class__.__name__ + f'(radius={self.radius})'


def transformation_composer(args, split: str = "train", mode: str = "Contrastive"):
    """Compose image transformation and augmentations

    Args: 
        args (argparse.ArgumentParser): command line arguments
        split (str): data split
        mode (str): training model, either CheXpert or Contrastive
            used to decide if TwoCropTransform should be applied 
    """

    if mode not in ["CheXpert", "Contrastive"]:
        raise Exception("Mode has to be either CheXpert or Contrastive")

    transforms = []
    transforms.append(t.Resize((args.resize_shape, args.resize_shape)))
    if split == "train":
        transforms.append(t.RandomCrop((args.crop_shape, args.crop_shape)))
        transforms.append(t.RandomRotation(
                                args.rotation_range, 
                                resample=PIL.Image.BICUBIC
                            ))
        # gaussian blur
        if args.gaussian_blur_radius is not None:
            transforms.append(GaussianBlur(args.gaussian_blur_radius))

    # to torch tensor
    transforms.append(t.ToTensor())

    # gaussian noise
    if split == "train":
        if (args.gaussian_noise_mean is not None) and (args.gaussian_noise_std) is not None:
            transforms.append(GaussianNoise(args.gaussian_noise_mean, args.gaussian_noise_std))

    # normalize
    transforms.append(t.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    transforms = t.Compose(transforms)

    # check if TwoCropTransform should be applied 
    if mode == "CheXpert":
        return transforms
    else:
        return TwoCropTransform(transforms)


def resize_img(img, scale):
    """
    Args:
        img - image as numpy array (cv2)
        scale - desired output image-size as scale x scale
    Return:
        image resized to scale x scale with shortest dimension 0-padded
    """
    size = img.shape
    max_dim = max(size)
    max_ind = size.index(max_dim)

    #Resizing
    if max_ind == 0:
        #image is heigher
        wpercent = (scale / float(size[0]))
        hsize = int((float(size[1]) * float(wpercent)))
        desireable_size = (scale, hsize)
    else:
        #image is wider
        hpercent = (scale / float(size[1]))
        wsize = int((float(size[0]) * float(hpercent)))
        desireable_size = (wsize, scale)
    resized_img = cv2.resize(img, desireable_size[::-1], interpolation = cv2.INTER_AREA) #this flips the desireable_size vector

    #Padding
    if max_ind == 0:
        # height fixed at scale, pad the width
        pad_size = scale - resized_img.shape[1]
        left = int(np.floor(pad_size/2))
        right = int(np.ceil(pad_size/2))
        top = int(0)
        bottom = int(0)
    else:
        # width fixed at scale, pad the height
        pad_size = scale - resized_img.shape[0]
        top = int(np.floor(pad_size/2))
        bottom = int(np.ceil(pad_size/2))
        left = int(0)
        right = int(0)
    resized_img = np.pad(resized_img,[(top, bottom), (left, right)], 'constant', constant_values=0)

    return resized_img

def unnormalize(img):
    """Un-normalize img with mean and std before logging on tensorboard"""
    invTrans = t.Compose([t.Normalize(mean = [ 0., 0., 0. ],
                                      std = 1 / np.array(IMAGENET_STD)),
                          t.Normalize(mean = - np.array(IMAGENET_MEAN),
                                      std = [ 1., 1., 1. ]),
                         ])
    img = invTrans(img)
    return img
                        
