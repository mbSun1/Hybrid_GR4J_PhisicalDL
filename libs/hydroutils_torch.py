import torch
import numpy as np

def nse_loss(y_true, y_pred):
    """
    计算Nash-Sutcliffe Efficiency (NSE)损失
    忽略前365天的数据
    """
    # 确保输入是torch张量
    y_true = torch.as_tensor(y_true, dtype=torch.float32)
    y_pred = torch.as_tensor(y_pred, dtype=torch.float32)
    
    # 忽略前365天的数据
    y_true = y_true[:, 365:, :]
    y_pred = y_pred[:, 365:, :]
    
    # 处理多变量情况
    if y_pred.shape[-1] == 6:  # 如果是analysis模式
        # 计算总流量
        qsub = y_pred[:, :, 4]  # 地下径流
        qsurf = y_pred[:, :, 5]  # 地表径流
        y_pred = qsub + qsurf
    
    # 计算NSE
    numerator = torch.sum((y_true - y_pred) ** 2, dim=[1, 2])
    denominator = torch.sum((y_true - torch.mean(y_true, dim=[1, 2], keepdim=True)) ** 2, dim=[1, 2])
    nse = 1 - numerator / (denominator + 1e-6)
    
    # 返回负NSE作为损失
    return -torch.mean(nse)

def nse_metrics(y_true, y_pred):
    """
    计算Nash-Sutcliffe Efficiency (NSE)指标
    忽略前365天的数据
    """
    # 确保输入是torch张量
    y_true = torch.as_tensor(y_true, dtype=torch.float32)
    y_pred = torch.as_tensor(y_pred, dtype=torch.float32)
    
    # 忽略前365天的数据
    y_true = y_true[:, 365:, :]
    y_pred = y_pred[:, 365:, :]
    
    # 处理多变量情况
    if y_pred.shape[-1] == 6:  # 如果是analysis模式
        # 计算总流量
        qsub = y_pred[:, :, 4]  # 地下径流
        qsurf = y_pred[:, :, 5]  # 地表径流
        y_pred = qsub + qsurf
    
    # 计算NSE
    numerator = torch.sum((y_true - y_pred) ** 2, dim=[1, 2])
    denominator = torch.sum((y_true - torch.mean(y_true, dim=[1, 2], keepdim=True)) ** 2, dim=[1, 2])
    nse = 1 - numerator / (denominator + 1e-6)
    
    return torch.mean(nse) 