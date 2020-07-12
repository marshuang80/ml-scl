"""Training script for CheXpert"""
import argparse
import numpy as np
import pandas as pd
import torch
import tqdm
import util

from constants        import *
from dataset.chexpert import get_dataloader 
from logger           import Logger
from models           import CheXpert


# Reproducibility
torch.manual_seed(6)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


def train(args):

    # get dataloader
    train_loader = get_dataloader(args, "train")
    valid_loader = get_dataloader(args, "valid")

    # get model and put on device 
    model = CheXpert(model_name="densenet121", num_classes=14)
    if args.device == "cuda" and len(args.gpu_ids) > 1:
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_type', type=str, default='Frontal', choices=['All','Frontal'])
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--iters_per_eval', type=int, default=100)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--num_epoch', type=int, default=3)
    parser.add_argument('--resize_shape', type=int, default=320)
    parser.add_argument('--crop_shape', type=int, default=320)
    parser.add_argument('--optimizer', type=str, default="adam", choices=["sgd", "adam", "adamw"])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--sgd_dampening', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--log_dir', type=str, default="./logs")
    parser.add_argument('--eval_metrics', type=str, default="auroc", choices=["auroc", "auprc", "accuracy", "precision", "recall"])

    args = parser.parse_args()

    # set gpu and device
    args.gpu_ids = [int(ids) for ids in args.gpu_ids.split(",")]
    if len(args.gpu_ids) > 0 and torch.cuda.is_available():
        # Set default GPU for `tensor.to('cuda')`
        torch.cuda.set_device(args.gpu_ids[0])
        args.device = 'cuda'
    else:
        args.device = 'cpu'

    train(args)