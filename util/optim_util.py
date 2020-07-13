import math
import numpy as np
import torch.optim as optim

from constants   import *


def adjust_learning_rate(args, optimizer, epoch):
    """Adjust learning rate

    Adapted from: 
        https://github.com/HobbitLong/SupContras
    """
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    """Warm up learning rate

    Adapted from:
       https://github.com/HobbitLong/SupContras 
    """
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    if opt.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), 
                              opt.lr,
                              momentum=opt.momentum,
                              weight_decay=opt.weight_decay,
                              dampening=opt.sgd_dampening)
    elif opt.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), 
                               opt.lr,
                               betas=(0.9, 0.999),
                               weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), 
                               opt.lr,
                               betas=(0.9, 0.999),
                               weight_decay=opt.weight_decay)
    else:
        raise ValueError(f'Unsupported optimizer: {opt.optimizer}')

    return optimizer