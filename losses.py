import torch
import torch.nn as nn

def symm_loss_1(preds, ys, emd_1, emd_2):
    mse = nn.MSELoss(reduction='mean')
    batch_loss = mse(ys, preds) + mse(emd_1, emd_2)
    return batch_loss

def symm_loss_2(preds, ys, lam=0.001):
    mse = nn.MSELoss(reduction='mean')
    batch_loss = mse(ys, preds) + lam * torch.pow(preds, 2)
    return batch_loss