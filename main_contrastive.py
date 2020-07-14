"""Training script for CheXpert"""
import argparse
import numpy as np
import pandas as pd
import torch
import tqdm
import util
import shutil

from constants        import *
from args             import ContrastiveTrainArgParser
from dataset.chexpert import get_dataloader 
from logger           import Logger
from models           import SupConModel
from eval.loss        import MultiClassSupConLoss


# Reproducibility
torch.manual_seed(6)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


def train(args):

    # get dataloader
    train_trainsforms = util.transformation_composer(args, "train", "Contrastive")
    train_loader = get_dataloader(args, train_trainsforms, "train", "Contrastive")

    # loss function 
    loss_fn = MultiClassSupConLoss(
        temperature = args.temp,
        contrast_mode = args.contrast_mode,
        match_type = args.match_type
    )

    # get model and put on device 
    model = SupConModel(
        model_name="densenet121",
        head=args.head,
        output_dim=args.output_dim
    )
    if args.device == "cuda":
        model = torch.nn.DataParallel(model, args.gpu_ids)

    model = model.to(args.device)
    loss_fn = loss_fn.to(args.device)


    # optimizer 
    optimizer = util.set_optimizer(opt=args, model=model)

    # define logger
    logger = Logger(
        log_dir=args.log_dir, 
        metrics_name="loss",
        args=args
    )

    # iterate over epoch
    global_step = 0
    min_loss = float("inf")
    accumulated_loss = []
    model.train()
    best_epoch = 0
    for epoch in range(args.num_epoch):

        # training loop
        for inputs, targets in tqdm.tqdm(train_loader, desc=f"[epoch {epoch}]"):

            with torch.set_grad_enabled(True):

                # stack labels and send to device
                targets = torch.stack(targets)
                targets = targets.transpose_(0,1)
                targets = targets.to(args.device, non_blocking=True)

                # stack inputs
                inputs = torch.cat([inputs[0], inputs[1]], dim=0)
                inputs = inputs.to(args.device, non_blocking=True)

                # compute loss
                features = model(inputs)
                batch_size = targets.shape[0]
                f1, f2 = torch.split(features, [batch_size, batch_size], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

                # Compute the minibatch loss.
                loss = loss_fn(features, targets.to(args.device))
                accumulated_loss.append(loss.item())

                logger.log_dict({"loss": loss}, global_step, "train")

                # Perform a backward pass.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # save checkpoint every N iterations 
            if global_step % args.iters_per_eval == 0:

                logger.log_image(inputs, global_step)
                avg_loss = np.mean(accumulated_loss)
                accumulated_loss = []

                print(min_loss, avg_loss)
            
                if avg_loss < min_loss:
                    best_epoch = epoch
                    min_loss = avg_loss
                    model_name = model.module.__class__.__name__

                    ckpt_dict = {
                        'model_name': model_name,
                        'model_args': model.module.cpu().args_dict(),
                        'model_state': model.module.cpu().state_dict()
                    }
                    ckpt_path = logger.ckpt_dir / f"{model_name}_{global_step}.pth"
                    torch.save(ckpt_dict, ckpt_path)
                    model = model.to(args.device)

            global_step += 1 
    
    # rename best checkpoint
    best_ckpt_path = str(logger.ckpt_dir / f"{model_name}_{best_epoch}.pth")
    new_ckpt_path = str(logger.ckpt_dir / f"{model_name}_best.pth")
    shutil.copyfile(best_ckpt_path, new_ckpt_path)

if __name__ == "__main__":

    # get argumetns 
    parser = ContrastiveTrainArgParser()
    args = parser.parse_args()
    train(args)
