import PIL
import torchvision.transforms as t
import cv2
import numpy as np
from constants   import *


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


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
                                resample=PIL.Image.BILINEAR
                            ))
    transforms.append(t.ToTensor())
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