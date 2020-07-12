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

        # dataset and augmentations
        self.parser.add_argument("--img_type", type=str, default="Frontal", choices=["All", "Frontal", "Lateral"])
        self.parser.add_argument("--uncertain", type=str, default="ignore", choices=["ignore", "zero", "one"])
        self.parser.add_argument("--resize_shape", type=int, default=320)
        self.parser.add_argument("--crop_shape", type=int, default=320)
        self.parser.add_argument("--rotation_range", type=int, default=20)

        # wandb dir
        self.parser.add_argument("--wandb_project_name", type=str, default="debug")

        # model
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
        return args