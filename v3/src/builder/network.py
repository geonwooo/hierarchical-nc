"""
모든 모델의 출력: (fine_100_logit, coarse_20_logit) 또는 fine_100_logit
모든 모델의 metric: fine_100 accuracy

차이점은 fine head에 들어가는 feature만:
  flat:               h 직접 → FC(100)
  sequential:         sg(p@W)        → per-group fine(5) → 100-way 조립
  sequential_residual: h + α·sg(p@W) → per-group fine(5) → 100-way 조립
  factorized:         h 직접          → per-group fine(5) → 100-way 조립
"""
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
            # A: h → FC(100) 직접
            self.classifier = self._get_classifier(self.num_classes)
            self.scaling = getattr(modules, cfg.scaling.type)(self.num_classes)

        else:
            # B/C/D/E/F 전부: coarse(20) + per-group fine(5) → 100-way
            # 차이는 classify()에서 fine head에 뭘 넣느냐만 다름
            self.classifier_coarse = self._get_classifier(self.num_classes_1)
            self.scaling_coarse = getattr(modules, cfg.scaling.type)(self.num_classes_1)
            f2c = self._get_fine_to_coarse(cfg)
            self.factorized_head = FactorizedPerGroup(
                self.num_features, self.num_classes_1, f2c)

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

        # === 아래 모델 전부: coarse + per-group fine → 100-way ===
        coarse_logits = self.scaling_coarse(self.classifier_coarse(h))

        # fine head에 들어갈 feature 결정 (유일한 차이점)
        if self.hier_type == 'sequential':
            # B: h를 버리고, coarse prediction으로 feature 재구성
            with torch.no_grad():
                prob = F.softmax(coarse_logits, dim=1)
                h_fine = prob @ self.classifier_coarse.weight  # sg(p@W)
        elif self.hier_type == 'sequential_residual':
            # E: h를 보존하고, coarse summary를 hint로만 더함
            with torch.no_grad():
                prob = F.softmax(coarse_logits, dim=1)
                summary = prob @ self.classifier_coarse.weight
            h_fine = h + self.residual_alpha * summary
        else:
            # C/D/F: h를 그대로 사용
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
        if cfg.dataset.random_hierarchy:
            import numpy as np
            rng = np.random.RandomState(cfg.seed_num)
            perm = rng.permutation(100)
            f2c = [0] * 100
            for g in range(20):
                for r in range(5):
                    f2c[perm[g * 5 + r]] = g
            return f2c
        from modules.hier_ops import FINE_TO_COARSE
        return FINE_TO_COARSE
