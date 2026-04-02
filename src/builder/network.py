import torch
import torch.nn as nn
import torch.nn.functional as F

import backbone
import modules
from modules.hier_ops import FactorizedPerGroup


class Network(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super(Network, self).__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.num_classes_1 = cfg.dataset.num_classes_1  # 0 means no hierarchy
        self.num_classes_2 = cfg.dataset.num_classes_2
        self.num_features = self.get_num_features()

        # --- hier_type determines the classify() behavior ---
        # 'default'              : legacy behavior (flat or sequential based on num_classes_1)
        # 'sequential'           : 선배님 original  (20→sg(Wp)→5, joint metric)
        # 'sequential_residual'  : residual variant (20→h+α·sg(Wp)→5)
        # 'factorized'           : coarse(20) + per-group fine(5) → 100-way
        self.hier_type = cfg.dataset.hier_type

        # Backward compat: if hier_type is 'default', fall back to old logic
        if self.hier_type == 'default':
            self.hierarchical = self.num_classes_1 > 0 and self.num_classes_2 > 0
            if self.hierarchical:
                self.hier_type = 'sequential'
            else:
                self.hier_type = 'flat'

        self.backbone = getattr(backbone, cfg.backbone.type)(cfg)
        self.pooling = getattr(modules, cfg.pooling.type)()
        self.reshape = getattr(modules, cfg.reshape.type)(cfg, num_features=self.num_features)

        # --- Build classifiers based on hier_type ---
        if self.hier_type in ('sequential', 'sequential_residual'):
            self.classifier_1 = self._get_classifier(self.num_classes_1)
            self.scaling_1 = getattr(modules, cfg.scaling.type)(self.num_classes_1)
            self.classifier_2 = self._get_classifier(self.num_classes_2)
            self.scaling_2 = getattr(modules, cfg.scaling.type)(self.num_classes_2)
            if self.hier_type == 'sequential_residual':
                self.residual_alpha = nn.Parameter(torch.tensor(0.2))

        elif self.hier_type == 'factorized':
            self.classifier_coarse = self._get_classifier(self.num_classes_1)
            self.scaling_coarse = getattr(modules, cfg.scaling.type)(self.num_classes_1)
            fine_to_coarse = self._get_fine_to_coarse(cfg)
            self.factorized_head = FactorizedPerGroup(
                self.num_features, self.num_classes_1, fine_to_coarse)

        else:  # 'flat'
            self.classifier = self._get_classifier(self.num_classes)
            self.scaling = getattr(modules, cfg.scaling.type)(self.num_classes)

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
        if self.hier_type == 'sequential':
            # 선배님 original: h → 20-way, sg(p@W) → 5-way
            output_1 = self.classifier_1(h)
            with torch.no_grad():
                prob_1 = F.softmax(output_1, dim=1)
                h2 = prob_1 @ self.classifier_1.weight   # (B, D)
            output_2 = self.classifier_2(h2)
            return self.scaling_1(output_1), self.scaling_2(output_2)

        elif self.hier_type == 'sequential_residual':
            # Residual variant: h2 = h + α·sg(p@W) instead of h2 = sg(p@W)
            output_1 = self.classifier_1(h)
            with torch.no_grad():
                prob_1 = F.softmax(output_1, dim=1)
                coarse_summary = prob_1 @ self.classifier_1.weight
            h2 = h + self.residual_alpha * coarse_summary
            output_2 = self.classifier_2(h2)
            return self.scaling_1(output_1), self.scaling_2(output_2)

        elif self.hier_type == 'factorized':
            # Coarse(20) + per-group fine(5) → assembled 100-way logits
            coarse_logits = self.scaling_coarse(self.classifier_coarse(h))
            fine_logits = self.factorized_head(h, coarse_logits)
            # Return (100-way, 20-way) — NOT same as sequential's (20-way, 5-way)
            # Trainer distinguishes by checking hier_type
            return fine_logits, coarse_logits

        else:  # flat
            return self.scaling(self.classifier(h))

    def get_num_features(self):
        dic_num_features = {
            'SimpleFNN': 300,
            'LeNet5': 84,
            'resnetcifar32_z': 64,
        }
        if self.cfg.backbone.type in dic_num_features:
            num_features = dic_num_features[self.cfg.backbone.type]
        elif 'vgg' in self.cfg.backbone.type:
            num_features = 4096
        else:
            raise NotImplementedError(
                "Update dic_num_features for {}".format(self.cfg.backbone.type))
        return num_features

    def _get_classifier(self, num_classes):
        bias_flag = self.cfg.classifier.bias
        if self.cfg.classifier.type == 'FC':
            return nn.Linear(self.num_features, num_classes, bias=bias_flag)
        else:
            return getattr(modules, self.cfg.classifier.type)(
                self.num_features, num_classes, cfg=self.cfg)

    @staticmethod
    def _get_fine_to_coarse(cfg):
        """Get fine→coarse mapping. Uses random split if random_hierarchy=True."""
        if cfg.dataset.random_hierarchy:
            import numpy as np
            rng = np.random.RandomState(cfg.seed_num)
            fine_perm = rng.permutation(100)
            fine_to_coarse = [0] * 100
            for level1_idx in range(20):
                for rank in range(5):
                    fine_to_coarse[fine_perm[level1_idx * 5 + rank]] = level1_idx
            return fine_to_coarse
        else:
            from modules.hier_ops import _FINE_TO_COARSE
            return _FINE_TO_COARSE
