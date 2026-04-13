import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

import backbone
import modules
from modules.hier_ops import FactorizedPerGroup, FINE_TO_COARSE


TRUE_SEQ_TYPES = {
    'true_seq_direct', 'true_seq_residual',
    'true_seq_probw', 'true_seq_probw_nosg',
}


def generate_etf(K, D):
    """Generate simplex ETF: K vertices in R^D, pairwise cosine = -1/(K-1)."""
    M = torch.randn(K, D)
    U, _, _ = torch.linalg.svd(M, full_matrices=False)
    M = U[:, :K]
    # Center
    M = M - M.mean(dim=0, keepdim=True)
    # Normalize rows
    M = F.normalize(M, dim=1)
    return M


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
                self.hier_type = 'true_seq_direct'
            else:
                self.hier_type = 'flat'

        self.is_true_sequential = (self.hier_type in TRUE_SEQ_TYPES)

        self.fine_hidden = getattr(cfg.loss, 'fine_hidden', 0)
        self.use_cosine = getattr(cfg.loss, 'cosine_classifier', False)
        self.cosine_scale = getattr(cfg.loss, 'cosine_scale', 16.0)
        self.use_film = getattr(cfg.loss, 'use_film', False)
        self.use_etf_coarse = getattr(cfg.loss, 'etf_coarse', False)
        self.use_etf_fine = getattr(cfg.loss, 'etf_fine', False)

        self.backbone = getattr(backbone, cfg.backbone.type)(cfg)
        self.pooling = getattr(modules, cfg.pooling.type)()
        self.reshape = getattr(modules, cfg.reshape.type)(
            cfg, num_features=self.num_features)

        # Bottleneck for D-sweep
        self.bottleneck_dim = getattr(cfg.loss, 'bottleneck_dim', 0)
        if self.bottleneck_dim > 0:
            self._orig_features = self.num_features
            self.bottleneck = nn.Linear(self.num_features, self.bottleneck_dim)
            self.num_features = self.bottleneck_dim

        if self.hier_type == 'flat':
            self.classifier = self._get_classifier(self.num_classes)
            self.scaling = getattr(modules, cfg.scaling.type)(self.num_classes)

        elif self.is_true_sequential:
            self._build_sequential(cfg)

        else:
            f2c = self._get_fine_to_coarse(cfg)
            num_coarse = len(set(f2c))
            self.classifier_coarse = self._get_classifier(num_coarse)
            self.scaling_coarse = getattr(modules, cfg.scaling.type)(num_coarse)
            self.factorized_head = FactorizedPerGroup(
                self.num_features, num_coarse, f2c)
            if self.hier_type == 'sequential_residual':
                self.residual_alpha = nn.Parameter(torch.tensor(0.2))

    def _build_sequential(self, cfg):
        D = self.num_features
        G = self.num_classes_1

        # Coarse classifier
        if self.use_etf_coarse:
            etf = generate_etf(G, D)
            self.register_buffer('coarse_etf', etf)
            self.coarse_scale = nn.Parameter(torch.tensor(16.0))
        elif self.use_cosine:
            self.coarse_weight = nn.Parameter(torch.empty(G, D))
            nn.init.kaiming_normal_(self.coarse_weight)
        else:
            self.classifier_coarse = self._get_classifier(G)
            self.scaling_coarse = getattr(modules, cfg.scaling.type)(G)

        # Fine structure
        f2c = self._get_fine_to_coarse(cfg)
        self.f2c = f2c
        groups = defaultdict(list)
        for c, g in enumerate(f2c):
            groups[g].append(c)
        max_fpg = max(len(groups[g]) for g in range(G))
        self.max_fpg = max_fpg

        # 100-way index buffers
        coarse_idx = torch.zeros(self.num_classes, dtype=torch.long)
        local_idx = torch.zeros(self.num_classes, dtype=torch.long)
        for g in range(G):
            for li, c in enumerate(sorted(groups[g])):
                coarse_idx[c] = g
                local_idx[c] = li
        self.register_buffer('coarse_idx', coarse_idx)
        self.register_buffer('local_idx', local_idx)

        # FiLM: per-group feature modulation
        if self.use_film:
            self.film_gamma = nn.Parameter(torch.ones(G, D))
            self.film_beta = nn.Parameter(torch.zeros(G, D))

        # Fine classifier
        if self.use_etf_fine:
            etf_fine = generate_etf(max_fpg, D)
            etf_all = etf_fine.unsqueeze(0).expand(G, -1, -1).clone()
            self.register_buffer('fine_etf', etf_all)
            self.fine_scale = nn.Parameter(torch.tensor(16.0))
        elif self.fine_hidden > 0:
            self.fine_w1 = nn.Parameter(
                torch.empty(D, G, self.fine_hidden))
            self.fine_b1 = nn.Parameter(torch.zeros(G, self.fine_hidden))
            self.fine_w2 = nn.Parameter(
                torch.empty(G, self.fine_hidden, max_fpg))
            self.fine_b2 = nn.Parameter(torch.zeros(G, max_fpg))
            nn.init.kaiming_normal_(self.fine_w1)
            nn.init.kaiming_normal_(self.fine_w2)
        else:
            self.fine_weight = nn.Parameter(
                torch.empty(D, G, max_fpg))
            self.fine_bias = nn.Parameter(torch.zeros(G, max_fpg))
            nn.init.kaiming_normal_(self.fine_weight)

        if self.hier_type == 'true_seq_residual':
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
        if self.bottleneck_dim > 0:
            x = self.bottleneck(x)
        return x

    def _coarse_forward(self, h):
        if self.use_etf_coarse:
            h_norm = F.normalize(h, dim=1)
            return self.coarse_scale * (h_norm @ self.coarse_etf.T)
        if self.use_cosine:
            h_norm = F.normalize(h, dim=1)
            w_norm = F.normalize(self.coarse_weight, dim=1)
            return self.cosine_scale * (h_norm @ w_norm.T)
        return self.scaling_coarse(self.classifier_coarse(h))

    def _fine_forward(self, h_fine):
        if self.use_etf_fine:
            h_norm = F.normalize(h_fine, dim=1)
            logits = self.fine_scale * torch.einsum(
                'bd,gfd->bgf', h_norm, self.fine_etf)
            return logits
        if self.fine_hidden > 0:
            hidden = F.relu(
                torch.einsum('bd,dgh->bgh', h_fine, self.fine_w1) + self.fine_b1)
            return torch.einsum('bgh,ghf->bgf', hidden, self.fine_w2) + self.fine_b2
        return torch.einsum('bd,dgf->bgf', h_fine, self.fine_weight) + self.fine_bias

    def _get_coarse_weight(self):
        if self.use_etf_coarse:
            return self.coarse_etf
        if self.use_cosine:
            return self.coarse_weight
        return self.classifier_coarse.weight

    def classify(self, h):
        if self.hier_type == 'flat':
            return self.scaling(self.classifier(h))

        if self.is_true_sequential:
            coarse_logits = self._coarse_forward(h)

            if self.hier_type == 'true_seq_direct':
                h_fine = h
            elif self.hier_type == 'true_seq_residual':
                with torch.no_grad():
                    prob = F.softmax(coarse_logits, dim=1)
                    summary = prob @ self._get_coarse_weight()
                h_fine = h + self.residual_alpha * summary
            elif self.hier_type == 'true_seq_probw':
                with torch.no_grad():
                    prob = F.softmax(coarse_logits, dim=1)
                    h_fine = prob @ self._get_coarse_weight()
            elif self.hier_type == 'true_seq_probw_nosg':
                prob = F.softmax(coarse_logits, dim=1)
                h_fine = prob @ self._get_coarse_weight()

            # FiLM: per-group feature modulation
            if self.use_film:
                B = h_fine.size(0)
                G = self.num_classes_1
                h_expanded = h_fine.unsqueeze(1).expand(B, G, -1)
                h_modulated = h_expanded * self.film_gamma.unsqueeze(0) + self.film_beta.unsqueeze(0)
                fine_all = self._fine_forward_film(h_modulated)
            else:
                fine_all = self._fine_forward(h_fine)

            return coarse_logits, fine_all

        # Factorized
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

    def _fine_forward_film(self, h_modulated):
        """Fine forward with FiLM-modulated per-group features. h_modulated: (B, G, D)"""
        if self.fine_hidden > 0:
            hidden = F.relu(
                torch.einsum('bgd,dgh->bgh', h_modulated, self.fine_w1) + self.fine_b1)
            return torch.einsum('bgh,ghf->bgf', hidden, self.fine_w2) + self.fine_b2
        return torch.einsum('bgd,dgf->bgf', h_modulated, self.fine_weight) + self.fine_bias

    def assemble_100way(self, coarse_logits, fine_all):
        return (coarse_logits[:, self.coarse_idx]
                + fine_all[:, self.coarse_idx, self.local_idx])

    def get_num_features(self):
        dic = {'SimpleFNN': 300, 'LeNet5': 84, 'resnetcifar32_z': 64}
        if self.cfg.backbone.type in dic:
            return dic[self.cfg.backbone.type]
        elif 'vgg' in self.cfg.backbone.type:
            return 4096
        raise NotImplementedError(
            "Update get_num_features for {}".format(self.cfg.backbone.type))

    def _get_classifier(self, num_classes):
        bias_flag = self.cfg.classifier.bias
        if self.cfg.classifier.type == 'FC':
            return nn.Linear(self.num_features, num_classes, bias=bias_flag)
        return getattr(modules, self.cfg.classifier.type)(
            self.num_features, num_classes, cfg=self.cfg)

    @staticmethod
    def _get_fine_to_coarse(cfg):
        grouping_file = getattr(cfg.dataset, 'grouping_file', 'none')
        if grouping_file and grouping_file != 'none':
            with open(grouping_file, 'r') as f:
                gdata = json.load(f)
            return gdata['fine_to_coarse']
        if cfg.dataset.random_hierarchy:
            import numpy as np
            rng = np.random.RandomState(cfg.seed_num)
            perm = rng.permutation(100)
            f2c = [0] * 100
            for g in range(20):
                for r in range(5):
                    f2c[perm[g * 5 + r]] = g
            return f2c
        return FINE_TO_COARSE
