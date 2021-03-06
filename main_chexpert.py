"""Training script for CheXpert"""
import argparse
import numpy as np
import pandas as pd
import torch
import tqdm
import util

from constants        import *
from args             import CheXpertTrainArgParser
from dataset.chexpert import get_dataloader 
from logger           import Logger
from models           import CheXpertModel


# Reproducibility
torch.manual_seed(6)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


def train(args):

    # get transformations
    train_trainsforms = util.transformation_composer(args, "train", "CheXpert")
    valid_trainsforms = util.transformation_composer(args, "valid", "CheXpert")

    # get dataloader
    train_loader = get_dataloader(args, train_trainsforms, "train", "CheXpert")
    valid_loader = get_dataloader(args, valid_trainsforms, "valid", "CheXpert")

    # get model and put on device 
    model = CheXpertModel(
        model_name=args.model_name, 
        num_classes=14
    )
    if args.device == "cuda":
        model = torch.nn.DataParallel(model, args.gpu_ids)
    model = model.to(args.device)

    # optimizer 
    optimizer = util.set_optimizer(opt=args, model=model)

    # loss function 
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")

    # define logger
    logger = Logger(
        log_dir=args.log_dir, 
        metrics_name=args.eval_metrics, 
        args=args
    )

    # iterate over epoch
    global_step = 0
    model.train()
    for epoch in range(args.num_epoch):

        # training loop
        for inputs, targets in tqdm.tqdm(train_loader, desc=f"[epoch {epoch}]"):

            # validate every N training iteration
            if global_step % args.iters_per_eval == 0:
                model.eval()
                probs, gt = [], []
                with torch.no_grad():

                    # validation loop
                    for val_inputs, val_targets in valid_loader:

                        batch_logits = model(val_inputs.to(args.device))
                        batch_probs = torch.sigmoid(batch_logits)

                        probs.append(batch_probs.cpu())
                        gt.append(val_targets.cpu())

                # evaluate results 
                metrics = util.evaluate(probs, gt, args.threshold)
                avg_metric, metric_dict = util.aggregate_metrics(metrics)

                # log image
                logger.log_iteration(metric_dict, global_step, "val")
                logger.log_dict(avg_metric, global_step, "val")
                logger.log_image(inputs, global_step)

                # save checkpoint
                logger.save_checkpoint(model, metrics, global_step)

                model.to(args.device)

            model.train()
            with torch.set_grad_enabled(True):

                # Run the minibatch through the model.
                logits = model(inputs.to(args.device))

                # Compute the minibatch loss.
                loss = loss_fn(logits, targets.to(args.device))

                logger.log_dict({"train/loss": loss}, global_step, "train")

                # Perform a backward pass.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            global_step += 1 


if __name__ == "__main__":

    # get argumetns 
    parser = CheXpertTrainArgParser()
    args = parser.parse_args()
    train(args)