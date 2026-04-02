import torch
import torch.nn as nn
from collections import defaultdict

__all__ = ['FactorizedPerGroup']

FINE_TO_COARSE = [
    4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
    3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
    6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
    0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
    5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
   16,  4, 17,  4,  2,  0, 17,  4, 18, 17,
   10,  3,  2, 12, 12, 16, 12,  1,  9, 19,
    2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
   16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
   18,  1,  2, 15,  6,  0, 17,  8, 14, 13,
]


class FactorizedPerGroup(nn.Module):
    """Per-group factorized head.
    logit[c] = coarse[g(c)] + fine_{g(c)}[l(c)]
    """
    def __init__(self, num_features, num_coarse=20, fine_to_coarse=None):
        super().__init__()
        if fine_to_coarse is None:
            fine_to_coarse = FINE_TO_COARSE

        groups = defaultdict(list)
        for c, g in enumerate(fine_to_coarse):
            groups[g].append(c)

        num_fine = len(fine_to_coarse)
        fpg = len(groups[0])
        coarse_idx = torch.zeros(num_fine, dtype=torch.long)
        local_idx = torch.zeros(num_fine, dtype=torch.long)
        for g in range(num_coarse):
            for li, c in enumerate(sorted(groups[g])):
                coarse_idx[c] = g
                local_idx[c] = li

        self.register_buffer('coarse_idx', coarse_idx)
        self.register_buffer('local_idx', local_idx)
        self.fine_weight = nn.Parameter(torch.empty(num_features, num_coarse, fpg))
        self.fine_bias = nn.Parameter(torch.zeros(num_coarse, fpg))
        nn.init.kaiming_normal_(self.fine_weight)

    def forward(self, h, coarse_logits):
        fine_all = torch.einsum('bd,dgf->bgf', h, self.fine_weight) + self.fine_bias
        return coarse_logits[:, self.coarse_idx] + fine_all[:, self.coarse_idx, self.local_idx]
