import sys
import os
import torch
import pytorch_lightning as pl
import numpy as np
import util
import wandb
sys.path.append(os.getcwd())

from eval            import *
from constants       import *
from argparse        import ArgumentParser
from models          import CheXpertModel
from dataset         import chexpert 


class LightningChexpertModel(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.loss = torch.nn.BCEWithLogitsLoss(reduction="mean")
        self.model = CheXpertModel(
            model_name = hparams.model_name,
            num_classes=14
        )

        # for computing metrics
        self.val_probs = []
        self.val_true = []

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        # compute loss
        loss = self.loss(y_hat, y)

        # logging
        result = pl.TrainResult(loss)
        result.log(
            'train_loss', loss, on_epoch=True, on_step=True, 
            sync_dist=True, logger=True, prog_bar=True
        )
        if batch_idx == 0:
            self.log_image(x)

        return result
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.type(torch.cuda.FloatTensor)
        y_hat = self.model(x)

        # compute loss
        loss = self.loss(y_hat, y)
        probs = torch.sigmoid(y_hat)

        # log loss
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.val_probs.append(probs.cpu().detach().numpy())
        self.val_true.append(y.cpu().detach().numpy())

        return result

    def validation_epoch_end(self, validation_result):
        # get metrics
        metrics = util.evaluate(self.val_probs, self.val_true, self.hparams.threshold)
        avg_metric, metric_dict = util.aggregate_metrics(metrics)

        # log metrics
        validation_result.log(
            'val_auroc', avg_metric['val_auroc'], on_epoch=True, sync_dist=True, logger=True
        )
        validation_result.log(
            'val_auprc', avg_metric['val_auprc'], on_epoch=True, sync_dist=True, logger=True
        )
        validation_result.val_loss = torch.mean(validation_result.val_loss)

        # log individual metrics 
        for key, val in metric_dict.items():
            validation_result.log(
                key, val, on_epoch=True, sync_dist=True, logger=True
            )

        # reset 
        self.val_probs = []
        self.val_true = []
        return validation_result

    def configure_optimizers(self):
        return util.set_optimizer(self.hparams, self.model)
    
    def log_image(self, x):
        image = x[0].cpu().numpy()
        self.logger.experiment[0].add_image('example',image, 0) 

    def __dataloader(self, split):

        transforms = util.transformation_composer(
            self.hparams, split, "CheXpert"
        )

        dataloader = chexpert.get_dataloader(
            args = self.hparams, 
            transforms = transforms,
            split = split,
            mode = "CheXpert"
        )
        return dataloader
    
    def train_dataloader(self):
        dataloader = self.__dataloader("train")
        return dataloader

    def val_dataloader(self):
        dataloader = self.__dataloader("valid")
        return dataloader

    def test_dataloader(self):
        dataloader = self.__dataloader("test")
        return dataloader 

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--threshold", type=float, default=0.5)
        parser.add_argument("--ckpt_path", type=str, default=None)
        return parser