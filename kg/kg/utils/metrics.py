# ===============================================
#
# Functions to compute metrics
#
# ===============================================
import logging
import os
import random
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    jaccard_score,
    precision_score,
    roc_auc_score,
)


def jaccard(y_true, y_pred):
    """Computes a Jaccard Coefficient"""

    score = []
    for b in range(y_true.shape[0]):
        target = np.where(y_true[b] == 1)[0]
        out_list = np.where(y_pred[b] == 1)[0]
        inter = set(out_list) & set(target)
        union = set(out_list) | set(target)
        jaccard_score = 0 if len(union) == 0 else len(inter) / len(union)
        score.append(jaccard_score)
    return np.mean(score)


def average_prc(y_true, y_pred):
    """Computes precision scores for each sample in batch"""
    score = []
    for b in range(y_true.shape[0]):
        target = np.where(y_true[b] == 1)[0]
        out_list = np.where(y_pred[b] == 1)[0]
        inter = set(out_list) & set(target)
        prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
        score.append(prc_score)
    return score


def average_recall(y_true, y_pred):
    """Computes recall scores for each sample in batch"""
    score = []
    for b in range(y_true.shape[0]):
        target = np.where(y_true[b] == 1)[0]
        out_list = np.where(y_pred[b] == 1)[0]
        inter = set(out_list) & set(target)
        recall_score = 0 if len(target) == 0 else len(inter) / len(target)
        score.append(recall_score)
    return score


def average_f1(average_prc, average_recall):
    """Computes f1 score for each sample in batch"""
    score = []
    for idx in range(len(average_prc)):
        if average_prc[idx] + average_recall[idx] == 0:
            score.append(0)
        else:
            score.append(
                2
                * average_prc[idx]
                * average_recall[idx]
                / (average_prc[idx] + average_recall[idx])
            )
    return score


def f1_metric(y_true, y_pred):
    """Computes f1 score for the entire batch"""
    all_micro = []
    for b in range(y_true.shape[0]):
        all_micro.append(f1_score(y_true[b], y_pred[b], average="macro"))
    return np.mean(all_micro)


def roc_auc(y_true, y_prob):
    """Computes average AuROC for the batch (per sample average)"""
    all_micro = []
    for b in range(len(y_true)):
        all_micro.append(roc_auc_score(y_true[b], y_prob[b], average="macro"))
    return np.mean(all_micro)


def precision_auc(y_true, y_prob):
    """Compute AuPR (estimate) for the batch (per sample average)"""
    # all_micro = []
    # for b in range(len(y_true)):
    #     all_micro.append(average_precision_score(
    #         y_true[b], y_prob[b], average='macro'))

    all_micro = [
        average_precision_score(y_t, y_p, average="macro") for y_t, y_p in zip(y_true, y_prob)
    ]

    return np.mean(all_micro)


def precision_at_k(y_true, y_prob, k=3):
    """Computes precision in retrieving `k` codes"""
    precision = 0
    sort_index = np.argsort(y_prob, axis=-1)[:, ::-1][:, :k]
    for i in range(len(y_true)):
        TP = 0
        for j in range(len(sort_index[i])):
            if y_true[i, sort_index[i, j]] == 1:
                TP += 1
        precision += TP / len(sort_index[i])
    return precision / len(y_true)


def multi_label_metric(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, fast: bool = False
) -> Tuple[float, ...]:
    """
    Returns a set of multi-label metrics

    Parameters
    ----------
    y_true: ground truth one-hot labels
    y_pred: thresholded predictions
    y_prob: sigmoid activated model output
    fast: compute only a subset of metrics

    Source: https://github.com/jshang123/G-Bert
    """

    # auc = roc_auc(y_true, y_prob)
    # p_1 = precision_at_k(y_true, y_prob, k=1)
    # p_3 = precision_at_k(y_true, y_prob, k=3)
    # p_5 = precision_at_k(y_true, y_prob, k=5)
    # f1 = f1_metric(y_true, y_pred)
    if not fast:
        prauc = precision_auc(y_true, y_prob)
        p_5 = precision_at_k(y_true, y_prob, k=5)

    ja = jaccard(y_true, y_pred)
    avg_prc = average_prc(y_true, y_pred)
    avg_recall = average_recall(y_true, y_pred)
    avg_f1 = average_f1(avg_prc, avg_recall)

    if not fast:
        return ja, prauc, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1), p_5
    else:
        return ja, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)


def metric_report(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.5,
    verbose: bool = False,
    fast: bool = False,
) -> Dict[str, float]:
    """
    Computes a set of metrics for multi-class one-hot vectors

    Parameters
    ----------
    y_pred: sigmoid activated model output
    y_true: one-hot ground truth labels
    threshold: for hard labels
    verbose: -
    fast: compute only a subset of metrics

    Source: https://github.com/jshang123/G-Bert
    """
    y_prob = y_pred.copy()
    y_pred[y_pred > threshold] = 1
    y_pred[y_pred <= threshold] = 0

    metric_container = {}
    if fast:
        ja, avg_p, avg_r, avg_f1 = multi_label_metric(y_true, y_pred, y_prob, fast=True)
        metric_container["jaccard"] = ja
        metric_container["f1"] = avg_f1
    else:
        ja, prauc, avg_p, avg_r, avg_f1, p_5 = multi_label_metric(y_true, y_pred, y_prob)
        metric_container["jaccard"] = ja
        metric_container["f1"] = avg_f1
        metric_container["prauc"] = prauc
        metric_container["average_recall"] = avg_r
        metric_container["average_precision"] = avg_p
        metric_container["precision_at_5"] = p_5

    if verbose:
        for k, v in metric_container.items():
            logging.info("%-10s : %-10.4f" % (k, v))

    return metric_container


def top_k_prec_recall(
    y_true_hot: np.ndarray, y_pred: np.ndarray, ks: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the precision and recall of the top-k most
    confident predictions of the network
    Reference: https://github.com/LuChang-CS/CGL/blob/b8b883395684b15f0b646628a7b3109aba82392a/metrics.py#L15

    Parameter
    ---------
    y_true_hot: np.ndarray
        ground truth one-hot encoded (Batch, Classes)
    y_pred: np.ndarray
        normalized confidence scores (Batch, Classes)
    ks: List[int]
        list of k's to compute metric at

    Return
    ------
    precision: np.ndarray
        Computed precision at levels k
    recall: np.ndarray
        Computed recall at levels k
    """
    a = np.zeros((len(ks),))
    r = np.zeros((len(ks),))

    # sort each sample of pred by confidence
    # and get indeces
    y_pred = -y_pred  # we can now sort ascending
    y_pred = np.argsort(y_pred, axis=-1)

    for pred, true_hot in zip(y_pred, y_true_hot):
        true = np.where(true_hot == 1)[0].tolist()
        t = set(true)
        for i, k in enumerate(ks):
            p = set(pred[:k])
            it = p.intersection(t)
            a[i] += len(it) / k
            r[i] += len(it) / len(t)

    precision = a / len(y_true_hot)
    recall = r / len(y_true_hot)

    return precision, recall
