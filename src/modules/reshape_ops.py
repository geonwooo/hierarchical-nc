import torch
import torch.nn as nn

__all__ = [
    'FlattenCustom',
]


class FlattenCustom(nn.Flatten):
    def __init__(self, cfg, **kwargs):
        start_dim = kwargs['start_dim'] if 'start_dim' in kwargs else 1
        end_dim = kwargs['end_dim'] if 'end_dim' in kwargs else -1
        super(FlattenCustom, self).__init__(start_dim=start_dim, end_dim=end_dim)

