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
        self.parser.add_argument("--gpu_ids", type=str, default='0')
        self.parser.add_argument("--num_workers", type=int, default=16)

        # training
        self.parser.add_argument("--lr", type=float, default=1e-3)
        self.parser.add_argument("--num_epoch", type=int, default=3)
        self.parser.add_argument("--batch_size", type=int, default=8)
        self.parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "adam", "adamw"])
        self.parser.add_argument("--lr_decay", type=float, default=0.1)
        self.parser.add_argument("--weight_decay", type=float, default=0.0)
        self.parser.add_argument("--momentum", type=float, default=0.9)
        self.parser.add_argument("--sgd_dampening", type=float, default=0.9)
        self.parser.add_argument("--iters_per_eval", type=int, default=100)

        # dataset and augmentations
        self.parser.add_argument("--img_type", type=str, default="Frontal", choices=["All", "Frontal", "Lateral"])
        self.parser.add_argument("--uncertain", type=str, default="ignore", choices=["ignore", "zero", "one"])
        self.parser.add_argument("--resize_shape", type=int, default=256)
        self.parser.add_argument("--crop_shape", type=int, default=256)
        self.parser.add_argument("--rotation_range", type=int, default=20)
        self.parser.add_argument("--gaussian_noise_mean", type=float, default=None)
        self.parser.add_argument("--gaussian_noise_std", type=float, default=None)
        self.parser.add_argument("--gaussian_blur_radius", type=float, default=None)

        # wandb dir
        self.parser.add_argument("--wandb_project_name", type=str, default="debug")

        # model
        self.parser.add_argument("--experiment_name", type=str, default="debug")
        self.parser.add_argument("--trial_suffix", type=str, default=None)
        self.parser.add_argument("--model_name", type=str, default="densenet121", choices=MODELS_2D.keys())

    def parse_args(self):
        args = self.parser.parse_args()
        
        # set gpu and device
        args.gpu_ids = [int(ids) for ids in args.gpu_ids.split(",")]
        args.num_gpus = len(args.gpu_ids)
        if len(args.gpu_ids) > 0 and torch.cuda.is_available():
            torch.cuda.set_device(args.gpu_ids[0])
            args.device = 'cuda'
        else:
            args.device = 'cpu'

        # scale batch size by number of gpus
        if args.device == "cuda":
            args.batch_size = args.batch_size * args.num_gpus

        # experiment name 
        args.experiment_name = f"{args.wandb_project_name}_{args.model_name}"
        if args.trial_suffix is None:
            args.trial_suffix = f"{args.optimizer}_lr{args.lr}_lrd{args.lr_decay}" + \
                                f"_wd{args.weight_decay}_rs{args.resize_shape}" + \
                                f"_cr{args.crop_shape}_ro{args.rotation_range}" + \
                                f"_gns{args.gaussian_noise_std}_gbr{args.gaussian_blur_radius}"
        
        args.experiment_name += f"_{args.trial_suffix}"

        return args
