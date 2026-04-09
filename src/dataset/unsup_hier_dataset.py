"""
UnsupHierCIFAR100: CIFAR-100 with hierarchy loaded from a JSON grouping file.

The JSON file is produced by tools/discover_hierarchy.py and contains:
  {"fine_to_coarse": [4, 1, 14, ...], "num_groups": 20, ...}

__getitem__ returns (img, fine_target, coarse_target, dummy_level2).
"""

import json
import numpy as np
from torchvision.datasets import CIFAR100


class UnsupHierCIFAR100(CIFAR100):
    """CIFAR100 with unsupervised 2-level hierarchy from JSON grouping file."""

    base_folder = "cifar-100-python"

    def __init__(
        self,
        cfg,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        **kwargs,
    ):
        super().__init__(cfg.dataset.root, train, transform, target_transform, download)

        grouping_file = cfg.dataset.grouping_file

        if grouping_file and grouping_file != 'none':
            with open(grouping_file, 'r') as f:
                gdata = json.load(f)
            fine_to_coarse = gdata['fine_to_coarse']
            self.num_groups = gdata['num_groups']
        else:
            # Fallback to CIFAR-100 ground truth hierarchy
            fine_to_coarse = [
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
            self.num_groups = 20

        self.fine_to_coarse = fine_to_coarse
        self.tgts_1 = [fine_to_coarse[t] for t in self.targets]
        self.tgts_2 = [0] * len(self.targets)  # dummy level-2

        # Build per-group fine index
        from collections import defaultdict
        self.groups = defaultdict(list)
        for c, g in enumerate(fine_to_coarse):
            self.groups[g].append(c)
        self.fine_per_group = len(self.groups[0])  # assume balanced

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, self.tgts_1[index], self.tgts_2[index]
