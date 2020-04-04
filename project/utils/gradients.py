"""
This file contains methods to get insights of the gradients during the training.
"""

import numpy as np

def get_gradient_norm2(m):
    """
    Computes the L2 Norm
    :param m: the model
    :return: the total norm for the model parameters
    """
    total_norm = 0
    for p in m.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def get_gradient_statistics(model):
    """
    Retrieves gradient statistic (DEPRECATED)
    :param model: model
    :return: min max mean value for the gradients
    """
    parameters = list(filter(lambda p: p.grad is not None, model.parameters()))
    min_stats = min(p.grad.data.min() for p in parameters)
    max_stats = max(p.grad.data.max() for p in parameters)
    mean_stats = np.mean([p.grad.mean().cpu().numpy() for p in parameters])

    return {"min": min_stats, "max": max_stats, "mean": mean_stats}