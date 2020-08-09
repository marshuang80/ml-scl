import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models as models_2d 
from typing      import Optional
from constants   import *


class SupConModel(nn.Module):
    """backbone + projection head
    
    Adapted from:
        https://github.com/HobbitLong/SupContrast
    """
    def __init__(
            self, 
            model_name='densenet121', 
            head='mlp', 
            output_dim=128
        ):
        super(SupConModel, self).__init__()

        # get model and feature dims 
        model_2d, features_dim = MODELS_2D[model_name]

        # set model variables
        self.output_dim = output_dim
        self.head = head
        self.model_name = model_name
        self.features_dim = features_dim
        self.encoder = model_2d()
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        # set output name
        if self.head == 'linear':
            self.fc = nn.Linear(self.features_dim, self.output_dim)
        elif self.head == 'mlp':
            self.fc = nn.Sequential(
                nn.Linear(self.features_dim, self.features_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.features_dim, self.output_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x, use_apex):
        #x = self.encoder.module.features(x) 
        with torch.cuda.amp.autocast(enabled=use_apex):
            x = self.encoder.features(x) 
            if self.model_name.startswith("dense"):
                x = F.relu(x, inplace=True)
            x = self.pool(x) 
            x = torch.flatten(x, 1)
            x = F.normalize(self.fc(x), dim=1)
        return x

    def args_dict(self):
        ckpt = {
            "model_name": self.model_name,
            "head": self.head,
            "output_dim": self.output_dim
        }
        return ckpt


class CheXpertModel(nn.Module):
    """Normal DenseNet for CheXpert"""
    def __init__(
            self, 
            model_name: str = 'densenet121', 
            num_classes: int = 14, 
            ckpt_path: Optional[str] = None, 
            imagenet_pretrain: bool = False
        ):
        super(CheXpertModel, self).__init__()

        self.model_name = model_name 
        self.num_classes = num_classes
        self.ckpt_path = ckpt_path
        self.imagenet_pretrain = imagenet_pretrain

        # Check if using contrastive pretrained 
        if self.ckpt_path is not None:
            ckpt = torch.load(self.ckpt_path)
            self.model = SupConModel(name=ckpt["model_args"].opt)
            self.model = nn.DataParallel(self.model)
            self.model.load_state_dict(ckpt["model_state"])
            features_dim = self.model.features_dim
        else:
            model_2d, features_dim = MODELS_2D[model_name]
            self.model = model_2d(pretrained=self.imagenet_pretrain)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.model.fc = nn.Linear(features_dim, self.num_classes)

    def forward(self, x):

        x = self.model.features(x)
        if self.model_name.startswith("dense"):
            x = F.relu(x, inplace=True)
        x = self.pool(x).view(x.size(0), -1)
        x = self.model.fc(x)
        return x

    def args_dict(self):
        ckpt = {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "ckpt_path": self.ckpt_path,
            "imagenet_pretrain": self.imagenet_pretrain
        }
        return ckpt