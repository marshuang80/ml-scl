import numpy as np
import torch
import pandas as pd
import sklearn.metrics as sk_metrics

from constants   import *


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    
    Args:
        output: prediction output
        target: target label
        topk: top k predictions to use for calculating accuracy 
    """
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


def undefined_catcher(func, x, y):
    """Return Nan if function returns undefined"""
    try:
        return func(x, y)
    except:
        return np.nan


def evaluate(probs, targets, threshold):
    """Compute several evaluation metrics. 

    Including: AUROC, AUPRC, accuracy, precision, recall 

    Args:
        probs (list): list of predicted probabilities
        targets (list): list of taget labels
        threshold (float): operating point used to threshold predictions
    """
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

def aggregate_metrics(metrics : dict):
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
        avg_metric[f"val_{m}"] = avg

    # save to log file
    metric_dict = {}
    for pathology, pathology_metrics in metrics.items():
        for metric, value in pathology_metrics.items():
            metric_dict[f"{pathology}_{metric}"] = value

    return avg_metric, metric_dict