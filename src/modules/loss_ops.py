import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np


class CrossEntropyCustom(nn.Module):
    def __init__(self, param_dict=None):
        super(CrossEntropyCustom, self).__init__()

    def forward(self, output, targets):
        return F.cross_entropy(output, targets)


# The LDAMLoss class is copied from the official PyTorch implementation in LDAM 
# (https://github.com/kaidic/LDAM-DRW)
class LDAMLoss(nn.Module):
    def __init__(self, param_dict=None):
        super(LDAMLoss, self).__init__()
        s = 30
        self.num_class_list = param_dict['num_class_list']
        self.rank = param_dict['rank']

        cfg = param_dict['cfg']
        max_m = cfg.loss.LDAM.max_margin
        m_list = 1.0 / np.sqrt(np.sqrt(self.num_class_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list).cuda(self.rank)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.step_epoch = cfg.loss.LDAM.drw_epoch
        self.weight = None

    def reset_epoch(self, epoch):
        idx = (epoch - 1) // self.step_epoch
        betas = [0, 0.9999]
        effective_num = 1.0 - np.power(betas[idx], self.num_class_list)
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.num_class_list)
        self.weight = torch.FloatTensor(per_cls_weights).cuda(self.rank)

    def forward(self, x, label):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, label.data.view(-1, 1), 1)

        index_float = index.type(torch.FloatTensor)
        index_float = index_float.cuda(self.rank)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, label, weight=self.weight)

