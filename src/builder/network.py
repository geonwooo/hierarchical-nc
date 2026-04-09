"""
v3 network.py — supports unsupervised hierarchy from JSON grouping file.
Only change from v3/src/builder/network.py: _get_fine_to_coarse loads from JSON.
"""
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

import backbone
import modules
from modules.hier_ops import FactorizedPerGroup, FINE_TO_COARSE


class Network(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super(Network, self).__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.num_classes_1 = cfg.dataset.num_classes_1
        self.num_classes_2 = cfg.dataset.num_classes_2
        self.num_features = self.get_num_features()

        self.hier_type = cfg.dataset.hier_type
        if self.hier_type == 'default':
            if self.num_classes_1 > 0 and self.num_classes_2 > 0:
                self.hier_type = 'sequential'
            else:
                self.hier_type = 'flat'

        self.backbone = getattr(backbone, cfg.backbone.type)(cfg)
        self.pooling = getattr(modules, cfg.pooling.type)()
        self.reshape = getattr(modules, cfg.reshape.type)(cfg, num_features=self.num_features)

        if self.hier_type == 'flat':
            self.classifier = self._get_classifier(self.num_classes)
            self.scaling = getattr(modules, cfg.scaling.type)(self.num_classes)
        else:
            f2c = self._get_fine_to_coarse(cfg)
            num_coarse = len(set(f2c))

            self.classifier_coarse = self._get_classifier(num_coarse)
            self.scaling_coarse = getattr(modules, cfg.scaling.type)(num_coarse)
            self.factorized_head = FactorizedPerGroup(
                self.num_features, num_coarse, f2c)

            if self.hier_type == 'sequential_residual':
                self.residual_alpha = nn.Parameter(torch.tensor(0.2))

    def forward(self, input, **kwargs):
        if 'feature_flag' in kwargs:
            return self.extract_feature(input)
        elif 'classifier_flag' in kwargs:
            return self.classify(input)
        return self.classify(self.extract_feature(input))

    def extract_feature(self, input):
        x = self.backbone(input)
        x = self.pooling(x)
        x = self.reshape(x)
        return x

    def classify(self, h):
        if self.hier_type == 'flat':
            return self.scaling(self.classifier(h))

        coarse_logits = self.scaling_coarse(self.classifier_coarse(h))

        if self.hier_type == 'sequential':
            with torch.no_grad():
                prob = F.softmax(coarse_logits, dim=1)
                h_fine = prob @ self.classifier_coarse.weight
        elif self.hier_type == 'sequential_residual':
            with torch.no_grad():
                prob = F.softmax(coarse_logits, dim=1)
                summary = prob @ self.classifier_coarse.weight
            h_fine = h + self.residual_alpha * summary
        else:
            h_fine = h

        fine_logits = self.factorized_head(h_fine, coarse_logits)
        return fine_logits, coarse_logits

    def get_num_features(self):
        dic = {'SimpleFNN': 300, 'LeNet5': 84, 'resnetcifar32_z': 64}
        if self.cfg.backbone.type in dic:
            return dic[self.cfg.backbone.type]
        elif 'vgg' in self.cfg.backbone.type:
            return 4096
        raise NotImplementedError(f"Update for {self.cfg.backbone.type}")

    def _get_classifier(self, num_classes):
        bias = self.cfg.classifier.bias
        if self.cfg.classifier.type == 'FC':
            return nn.Linear(self.num_features, num_classes, bias=bias)
        return getattr(modules, self.cfg.classifier.type)(
            self.num_features, num_classes, cfg=self.cfg)

    @staticmethod
    def _get_fine_to_coarse(cfg):
        """Load fine→coarse mapping. Priority:                          # CHANGED
        1. JSON grouping file (unsupervised)                            # CHANGED
        2. Random split (random_hierarchy=True)                         # CHANGED
        3. CIFAR-100 GT coarse labels (default)                         # CHANGED
        """
        # 1. JSON grouping file                                         # CHANGED
        grouping_file = getattr(cfg.dataset, 'grouping_file', 'none')   # CHANGED
        if grouping_file and grouping_file != 'none':                   # CHANGED
            with open(grouping_file, 'r') as f:                         # CHANGED
                gdata = json.load(f)                                    # CHANGED
            return gdata['fine_to_coarse']                              # CHANGED

        # 2. Random split
        if cfg.dataset.random_hierarchy:
            import numpy as np
            rng = np.random.RandomState(cfg.seed_num)
            perm = rng.permutation(100)
            f2c = [0] * 100
            for g in range(20):
                for r in range(5):
                    f2c[perm[g * 5 + r]] = g
            return f2c

        # 3. GT
        return FINE_TO_COARSE
