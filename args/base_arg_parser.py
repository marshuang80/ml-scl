import os
import sys
import torch
import argparse
sys.path.append(os.getcwd())

from constants import *


class BaseTrainArgParser:
    '''Base training argument parser
    Shared with CheXpert and Contrastive loss Training
    '''

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description = "Training Arguments for constrastive learning")
        
        # hardware stepup
        self.parser.add_argument("--wandb_project_name", type=str, default="CheXpert")
        self.parser.add_argument("--log_dir", type=str, default="/data4/selfsupervision/log/chexpert")

        # training
        self.parser.add_argument("--lr", type=float, default=1e-3)
        self.parser.add_argument("--batch_size", type=int, default=128)
        self.parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "adam", "adamw"])
        self.parser.add_argument("--lr_decay", type=float, default=0.1)
        self.parser.add_argument("--weight_decay", type=float, default=0.0)
        self.parser.add_argument("--momentum", type=float, default=0.9)
        self.parser.add_argument("--sgd_dampening", type=float, default=0.9)

        # dataset and augmentations
        self.parser.add_argument("--num_workers", type=int, default=16)
        self.parser.add_argument("--img_type", type=str, default="Frontal", choices=["All", "Frontal", "Lateral"])
        self.parser.add_argument("--uncertain", type=str, default="ignore", choices=["ignore", "zero", "one"])
        self.parser.add_argument("--resize_shape", type=int, default=256)
        self.parser.add_argument("--crop_shape", type=int, default=224)
        self.parser.add_argument("--rotation_range", type=int, default=20)
        self.parser.add_argument("--gaussian_noise_mean", type=float, default=None)
        self.parser.add_argument("--gaussian_noise_std", type=float, default=None)
        self.parser.add_argument("--gaussian_blur_radius", type=float, default=None)

        # model
        self.parser.add_argument("--experiment_name", type=str, default="debug")
        self.parser.add_argument("--trial_suffix", type=str, default=None)
        self.parser.add_argument("--model_name", type=str, default="densenet121", choices=MODELS_2D.keys())

    def get_parser(self):
        return self.parser

    def parse_args(self):
        args = self.parser.parse_args()
        
        # experiment name 
        args.experiment_name = f"{args.wandb_project_name}_{args.model_name}"
        if args.trial_suffix is None:
            args.trial_suffix = f"{args.optimizer}_lr{args.lr}_lrd{args.lr_decay}" + \
                                f"_wd{args.weight_decay}_rs{args.resize_shape}" + \
                                f"_cr{args.crop_shape}_ro{args.rotation_range}" + \
                                f"_gns{args.gaussian_noise_std}_gbr{args.gaussian_blur_radius}"
        
        args.experiment_name += f"_{args.trial_suffix}"

        return args

    def str2bool(self, v):
        """convert input argument to bool"""
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
