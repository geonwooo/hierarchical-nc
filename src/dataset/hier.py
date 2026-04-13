import numpy as np
from collections import defaultdict
from torchvision.datasets import CIFAR100


class HierCIFAR100(CIFAR100):
    """CIFAR100 with two-level hierarchical labels.

    __getitem__ returns (img, fine_target, coarse_target, local_fine_target).
    tgts_1: coarse group index (0-19)
    tgts_2: local fine index within the coarse group (0-4)
    """

    base_folder = "cifar-100-python"

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

    def __init__(self, cfg, train=True, transform=None,
                 target_transform=None, download=False, **kwargs):
        super().__init__(
            cfg.dataset.root, train, transform,
            target_transform, download)

        if cfg.dataset.random_hierarchy:
            fine_to_coarse = self._random_f2c(cfg.seed_num)
        else:
            fine_to_coarse = self.FINE_TO_COARSE

        groups = defaultdict(list)
        for c, g in enumerate(fine_to_coarse):
            groups[g].append(c)

        fine_to_local = {}
        for g in groups:
            for li, c in enumerate(sorted(groups[g])):
                fine_to_local[c] = li

        self.fine_to_coarse = fine_to_coarse
        self.fine_to_local = fine_to_local
        self.tgts_1 = [fine_to_coarse[t] for t in self.targets]
        self.tgts_2 = [fine_to_local[t] for t in self.targets]

    @staticmethod
    def _random_f2c(seed):
        rng = np.random.RandomState(seed)
        perm = rng.permutation(100)
        f2c = [0] * 100
        for g in range(20):
            for r in range(5):
                f2c[perm[g * 5 + r]] = g
        return f2c

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, self.tgts_1[index], self.tgts_2[index]
