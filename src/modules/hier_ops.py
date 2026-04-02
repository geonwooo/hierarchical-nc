"""
Hierarchical classifier modules for NC × LM pilot experiments.
Drop this file into src/modules/ of the senior's codebase.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['FactorizedPerGroup']

# CIFAR-100 fine→coarse mapping (same as HierCIFAR100.FINE_TO_COARSE)
_FINE_TO_COARSE = [
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


def _build_mapping(fine_to_coarse, num_coarse=20):
    """Build fine_class → (coarse_group, within_group_index) mapping."""
    from collections import defaultdict
    coarse_to_fine = defaultdict(list)
    for fine_cls, coarse_cls in enumerate(fine_to_coarse):
        coarse_to_fine[coarse_cls].append(fine_cls)

    num_fine = len(fine_to_coarse)
    coarse_idx = torch.zeros(num_fine, dtype=torch.long)
    local_idx = torch.zeros(num_fine, dtype=torch.long)

    for coarse_cls in range(num_coarse):
        fine_list = sorted(coarse_to_fine[coarse_cls])
        for local_i, fine_cls in enumerate(fine_list):
            coarse_idx[fine_cls] = coarse_cls
            local_idx[fine_cls] = local_i

    num_fine_per_group = len(coarse_to_fine[0])  # assume uniform (=5 for CIFAR-100)
    return coarse_idx, local_idx, num_fine_per_group


class FactorizedPerGroup(nn.Module):
    """Per-group factorized classifier.

    For each coarse group g, a separate fine-level weight matrix W_g.
    Final logit for fine class c:
        logit[c] = coarse_logit[g(c)] + fine_logit_{g(c)}[f(c)]

    where g(c) = coarse group of c, f(c) = within-group index of c.

    Args:
        num_features: backbone feature dim (e.g., 4096 for VGG11)
        num_coarse: number of coarse groups (20)
        fine_to_coarse: list mapping fine_class → coarse_class
    """

    def __init__(self, num_features, num_coarse=20, fine_to_coarse=None):
        super().__init__()
        if fine_to_coarse is None:
            fine_to_coarse = _FINE_TO_COARSE

        coarse_idx, local_idx, num_fine_per_group = _build_mapping(
            fine_to_coarse, num_coarse)

        self.num_coarse = num_coarse
        self.num_fine_per_group = num_fine_per_group
        self.num_fine_total = len(fine_to_coarse)

        # Register as buffers (not parameters, but move to GPU with model)
        self.register_buffer('coarse_idx', coarse_idx)
        self.register_buffer('local_idx', local_idx)

        # Per-group fine weight: (num_features, num_coarse, num_fine_per_group)
        # e.g., (4096, 20, 5) for VGG11 + CIFAR-100
        self.fine_weight = nn.Parameter(
            torch.empty(num_features, num_coarse, num_fine_per_group))
        self.fine_bias = nn.Parameter(
            torch.zeros(num_coarse, num_fine_per_group))

        # Init
        nn.init.kaiming_normal_(self.fine_weight)

    def forward(self, h, coarse_logits):
        """
        Args:
            h: (B, D) hidden features
            coarse_logits: (B, num_coarse) coarse classification logits

        Returns:
            logits_fine: (B, num_fine_total) full fine-grained logits
        """
        # Per-group fine logits: (B, num_coarse, num_fine_per_group)
        fine_all = torch.einsum('bd,dgf->bgf', h, self.fine_weight) + self.fine_bias

        # Assemble 100-way logits using vectorized indexing
        # coarse_idx[c] = g(c), local_idx[c] = f(c)
        # logit[c] = coarse_logits[:, g(c)] + fine_all[:, g(c), f(c)]
        logits_fine = (
            coarse_logits[:, self.coarse_idx]
            + fine_all[:, self.coarse_idx, self.local_idx]
        )

        return logits_fine

    def extra_repr(self):
        return (f'num_coarse={self.num_coarse}, '
                f'num_fine_per_group={self.num_fine_per_group}, '
                f'total_fine={self.num_fine_total}')
