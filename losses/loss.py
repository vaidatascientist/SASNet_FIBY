import torch
import torch.nn as nn

def calculate_bce_loss(pred_density, gt_density):
    loss = nn.BCELoss(pred_density, gt_density)
    return 10*loss


def calculate_euclidean_loss(pred_density, gt_density):
    loss = torch.sqrt(torch.mean((pred_density - gt_density) ** 2))
    return 3*loss
