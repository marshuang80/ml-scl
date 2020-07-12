import math
import numpy as np
import torch
import torch.optim as optim
import cv2
import pandas as pd
import sklearn.metrics as sk_metrics

from constants import *


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
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


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

def undefined_catcher(func, x, y):
    try:
        return func(x, y)
    except:
        return np.nan

def evaluate(probs, targets, threshold):
    # aggregate results
    probs_concat = np.concatenate(probs)
    gt_concat = np.concatenate(targets)
    probs_df = pd.DataFrame({task: probs_concat[:, i]
                        for i, task in enumerate(CHEXPERT_TASKS)})
    gt_df = pd.DataFrame({task: gt_concat[:, i]
                    for i, task in enumerate(CHEXPERT_TASKS)})
    preds_df = {}
    for i, task in enumerate(CHEXPERT_TASKS):
        pred = [1 if p >= threshold else 0 for p in probs_concat[:,i]]
        preds_df[task] = pred

    # loop over tasks
    metrics = dict()
    for task in CHEXPERT_TASKS:
        # extract task specific predictions and label
        task_gt = gt_df[task]
        task_probs = probs_df[task]
        task_preds = preds_df[task]

        # calculate metrics 
        tasks_metrics = dict()
        tasks_metrics['auprc'] = undefined_catcher(sk_metrics.average_precision_score, task_gt, task_probs)
        tasks_metrics['auroc'] = undefined_catcher(sk_metrics.roc_auc_score, task_gt, task_probs)
        tasks_metrics['accuracy'] = undefined_catcher(sk_metrics.accuracy_score, task_gt, task_preds)
        tasks_metrics['precision'] = undefined_catcher(sk_metrics.precision_score, task_gt, task_preds)
        tasks_metrics['recall'] = undefined_catcher(sk_metrics.recall_score, task_gt, task_preds)

        metrics[task] = tasks_metrics
    
    return metrics

def aggregate_metrics(metrics : dict ):
    """Aggregate evaluation metrics 

    Args:
        metrics (dict): dictionary of evaluation metrics for each output class
    """

    # log into tensorboard 
    avg_metric = {}
    for m in CHEXPERT_EVAL_METRICS:
        metrics_list = []
        for pathology in CHEXPERT_COMPETITION_TASKS:
            metrics_list.append(metrics[pathology][m])
        
        avg = sum(metrics_list) / len(metrics_list)
        avg_metric[f"val/{m}"] = avg

    # save to log file
    metric_dict = {}
    for pathology, pathology_metrics in metrics.items():
        for metric, value in pathology_metrics.items():
            metric_dict[f"{pathology}_{metric}"] = [value]

    return avg_metric, metric_dict

def resize_img(img, scale):
    """
    args:
        img - image as numpy array (cv2)
        scale - desired output image-size as scale x scale
    return:
        image resized to scale x scale with shortest dimension 0-padded
    """
    size = img.shape
    max_dim = max(size)
    max_ind = size.index(max_dim)

    #Resizing
    if max_ind == 0:
        #image is heigher
        wpercent = (scale / float(size[0]))
        hsize = int((float(size[1]) * float(wpercent)))
        desireable_size = (scale, hsize)
    else:
        #image is wider
        hpercent = (scale / float(size[1]))
        wsize = int((float(size[0]) * float(hpercent)))
        desireable_size = (wsize, scale)
    resized_img = cv2.resize(img, desireable_size[::-1], interpolation = cv2.INTER_AREA) #this flips the desireable_size vector

    #Padding
    if max_ind == 0:
        # height fixed at scale, pad the width
        pad_size = scale - resized_img.shape[1]
        left = int(np.floor(pad_size/2))
        right = int(np.ceil(pad_size/2))
        top = int(0)
        bottom = int(0)
    else:
        # width fixed at scale, pad the height
        pad_size = scale - resized_img.shape[0]
        top = int(np.floor(pad_size/2))
        bottom = int(np.ceil(pad_size/2))
        left = int(0)
        right = int(0)
    resized_img = np.pad(resized_img,[(top, bottom), (left, right)], 'constant', constant_values=0)

    return resized_img