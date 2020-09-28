import sys
import os
import torch
import pytorch_lightning as pl
import numpy as np
import util
import wandb
sys.path.append(os.getcwd())

from eval.loss import MultiClassSupConLoss
from constants import *
from argparse  import ArgumentParser
from models    import SupConModel
from dataset   import chexpert 


class LightningContrastiveModel(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.loss_fn = MultiClassSupConLoss(
            temperature = hparams.temp,
            contrast_mode = hparams.contrast_mode,
            match_type = hparams.match_type
        )
        self.model = SupConModel(
            model_name = hparams.model_name,
            head = hparams.head,
            output_dim = hparams.output_dim
        )

    def training_step(self, batch, batch_idx):

        # stack input and targets 
        inputs, targets = batch
        targets = torch.stack(targets)
        targets = targets.transpose_(0,1)
        inputs = torch.cat([inputs[0], inputs[1]], dim=0)

        # get features 
        features = self.model(inputs)

        # compute loss
        batch_size = targets.shape[0]
        f1, f2 = torch.split(features, [batch_size, batch_size], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = self.loss_fn(features, targets)

        # logging
        result = pl.TrainResult(loss)
        result.log(
            'train_loss', loss, on_epoch=True, on_step=True, 
            sync_dist=True, logger=True, prog_bar=True
        )
        if batch_idx == 0:
            self.log_image(inputs)

        return result
        
    def configure_optimizers(self):
        return util.set_optimizer(self.hparams, self.model)
    
    def log_image(self, x):
        image = x[0].cpu().numpy()
        self.logger.experiment[0].add_image('example',image, 0)

    def __dataloader(self, split):

        transforms = util.transformation_composer(
            self.hparams, split, "Contrastive"
        )

        dataloader = chexpert.get_dataloader(
            args = self.hparams, 
            transforms = transforms,
            split = split,
            mode = "Contrastive"
        )
        return dataloader
    
    def train_dataloader(self):
        dataloader = self.__dataloader("train")
        return dataloader

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--temp", type=float, default=0.07)
        parser.add_argument("--head", type=str, default="mlp", choices=["mlp", "linear"])
        parser.add_argument("--output_dim", type=int, default=128)
        parser.add_argument("--contrast_mode", type=str, default="all", choices=["one", "all"])
        parser.add_argument("--match_type", type=str, default="all", 
                            choices=["all", "any", "iou_weighted", "f1_weighted", "one_weighted", "zero_and_one_weighted"])
        return parser