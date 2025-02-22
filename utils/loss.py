import torch
import torch.nn as nn
import numpy as np

class mse_loss(nn.Module):
    def __init__(self):

        super(mse_loss, self).__init__()


    def forward(self, y_true, y_pred, skip_index=0, batch_first=True):
        """
        y_true & y_pred: torch.Tensor, shape:(batch size, sequence length, output dim) or (sequence length, output_dim)
        skip_index: int, if a warm-up period is adopted it should be the length of the warm-up
        """
        device = y_pred.device
        assert y_true.dim() in [2, 3]
        assert y_pred.dim() in [2, 3]

        # convert the shape of (sequence length, output_dim) to (batch size, sequence length, output dim)
        batch_dim = 0 if batch_first else 1
        if y_true.dim() == 2:
            y_true = y_true.unsqueeze(batch_dim)
        if y_pred.dim() == 2:
            y_pred = y_pred.unsqueeze(batch_dim)

        # Omit values in the warm-up period
        y_true = y_true[:, skip_index:]
        y_pred = y_pred[:, skip_index:]

        # pad with 0 for nan values in the observation and then mask the 0 values in later calculation
        mask = torch.where(torch.isnan(y_true), torch.zeros_like(y_true), torch.ones_like(y_true))
        y_true = torch.where(torch.isnan(y_true), torch.zeros_like(y_true), y_true)

        # calculate mse for the daily streamflow
        mse_daily = (((y_pred - y_true) ** 2) * mask).sum() / mask.sum()

        return mse_daily


class nse_loss(nn.Module):
    def __init__(self):
        super(nse_loss, self).__init__()


    def forward(self, y_true, y_pred, skip_index=0, batch_first=True):
        return 1 - nse(y_true, y_pred, skip_index, batch_first)


def nse(y_true, y_pred, skip_index, batch_first=True):
    """
    y_true & y_pred: torch.Tensor, shape:(batch size, sequence length, output dim) or (sequence length, output_dim)
    skip_index: int, if a warm-up period is adopted it should be the length of the warm-up
    """
    device = y_pred.device
    assert y_true.dim() in [2, 3]
    assert y_pred.dim() in [2, 3]

    # convert the shape of (sequence length, output_dim) to (batch size, sequence length, output dim)
    batch_dim = 0 if batch_first else 1
    if y_true.dim() == 2:
        y_true = y_true.unsqueeze(batch_dim)
    if y_pred.dim() == 2:
        y_pred = y_pred.unsqueeze(batch_dim)

    # Omit values in the warm-up period
    y_true = y_true[:, skip_index:]
    y_pred = y_pred[:, skip_index:]

    # calculate daily mean streamflow
    seq_dim = 1 if batch_first else 0
    y_true_mean = torch.nanmean(y_true, dim=seq_dim, keepdim=True)

    # pad with 0 for nan values in the observation and then mask the 0 values in later calculation
    mask = torch.where(torch.isnan(y_true), torch.zeros_like(y_true), torch.ones_like(y_true))
    y_true = torch.where(torch.isnan(y_true), torch.zeros_like(y_true), y_true)

    # calculate nse for the daily streamflow
    nse_daily = (1 - ((y_pred - y_true) ** 2 * mask).sum(dim=seq_dim, keepdim=True) /
                 ((y_true - y_true_mean) ** 2 * mask).sum(dim=seq_dim, keepdim=True)).mean()


    return nse_daily


