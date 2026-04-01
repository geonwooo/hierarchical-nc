import numpy as np
from torchvision.datasets import CIFAR100


class HierCIFAR100(CIFAR100):
    """CIFAR100 with two-level hierarchical labels.

    Two hierarchy modes are available via cfg.dataset.random_hierarchy:

    Fixed (random_hierarchy=False):
        Level 1 (tgts_1, 20 classes): standard CIFAR100 coarse labels.
            0: aquatic_mammals       1: fish
            2: flowers               3: food_containers
            4: fruit_and_vegetables  5: household_electrical_devices
            6: household_furniture   7: insects
            8: large_carnivores      9: large_man-made_outdoor_things
           10: large_natural_outdoor_scenes
           11: large_omnivores_and_herbivores
           12: medium_mammals        13: non-insect_invertebrates
           14: people                15: reptiles
           16: small_mammals         17: trees
           18: vehicles_1            19: vehicles_2

        Level 2 (tgts_2, 5 classes): semantic grouping of the 20 coarse classes.
            0: aquatic/water animals  — coarse {0, 1, 13, 15}
            1: land/air animals       — coarse {7, 8, 12, 16}
            2: nature & plants        — coarse {2, 10, 11, 17}
            3: people & food          — coarse {3, 4, 5, 14}
            4: man-made               — coarse {6, 9, 18, 19}

    Random (random_hierarchy=True):
        Level 1 (tgts_1, 20 classes): 100 fine classes randomly partitioned
            into 20 groups of 5, using cfg.seed_num as the RNG seed.
        Level 2 (tgts_2, 5 classes): 20 level-1 classes randomly partitioned
            into 5 groups of 4, continuing with the same RNG state.

    __getitem__ returns (img, target, tgts_1, tgts_2).
    """

    base_folder = "cifar-100-python"

    # Ground-truth fine→coarse mapping from the CIFAR100 meta file (indexed by fine label 0-99).
    FINE_TO_COARSE = [
        4,  1, 14,  8,  0,  6,  7,  7, 18,  3,   # 0-9
        3, 14,  9, 18,  7, 11,  3,  9,  7, 11,   # 10-19
        6, 11,  5, 10,  7,  6, 13, 15,  3, 15,   # 20-29
        0, 11,  1, 10, 12, 14, 16,  9, 11,  5,   # 30-39
        5, 19,  8,  8, 15, 13, 14, 17, 18, 10,   # 40-49
       16,  4, 17,  4,  2,  0, 17,  4, 18, 17,   # 50-59
       10,  3,  2, 12, 12, 16, 12,  1,  9, 19,   # 60-69
        2, 10,  0,  1, 16, 12,  9, 13, 15, 13,   # 70-79
       16, 19,  2,  4,  6, 19,  5,  5,  8, 19,   # 80-89
       18,  1,  2, 15,  6,  0, 17,  8, 14, 13,   # 90-99
    ]

    # Coarse→level-2 mapping (indexed by coarse label 0-19).
    COARSE_TO_LEVEL2 = [
        0, 0, 2, 3, 3, 3, 4, 1, 1, 4,   # coarse 0-9
        2, 2, 1, 0, 3, 0, 1, 2, 4, 4,   # coarse 10-19
    ]

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

        if cfg.dataset.random_hierarchy:
            fine_to_coarse, coarse_to_level2 = self._random_split(cfg.seed_num)
        else:
            fine_to_coarse = self.FINE_TO_COARSE
            coarse_to_level2 = self.COARSE_TO_LEVEL2

        self.tgts_1 = [fine_to_coarse[t] for t in self.targets]
        self.tgts_2 = [coarse_to_level2[c] for c in self.tgts_1]

    @staticmethod
    def _random_split(seed):
        """Randomly partition 100 fine classes into 20 level-1 groups (5 each),
        then partition 20 level-1 classes into 5 level-2 groups (4 each).
        The same seed yields the same split for both train and test sets.
        """
        rng = np.random.RandomState(seed)

        fine_perm = rng.permutation(100)
        fine_to_coarse = [0] * 100
        for level1_idx in range(20):
            for rank in range(5):
                fine_to_coarse[fine_perm[level1_idx * 5 + rank]] = level1_idx

        coarse_perm = rng.permutation(20)
        coarse_to_level2 = [0] * 20
        for level2_idx in range(5):
            for rank in range(4):
                coarse_to_level2[coarse_perm[level2_idx * 4 + rank]] = level2_idx

        return fine_to_coarse, coarse_to_level2

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, self.tgts_1[index], self.tgts_2[index]
