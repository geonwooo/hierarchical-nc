"""
Extended trainer for unsupervised hierarchy experiments.
Adds: dual temperature, NC-aware regularizer, λ warm-up.

Drop-in replacement for v3/src/core/trainer.py
New config keys used:
  loss.dual_temp_coarse: float (default 1.0, higher = softer coarse decisions)
  loss.dual_temp_fine: float (default 1.0, lower = sharper fine decisions)
  loss.nc_reg_weight: float (default 0.0, >0 enables NC-aware regularizer)
  loss.lambda_warmup: bool (default False, True = λ anneals from 1.0 to lambda_coarse)
"""

import torch
import math
import numpy as np
from collections import defaultdict
from contextlib import nullcontext
import torch.nn.functional as F

import utils.mixup_utils as mixup_utils
from core.evaluate import accuracy


class Trainer:
    def __init__(self, cfg, rank):
        self.cfg = cfg
        self.type = cfg.train.trainer.type
        self.rank = rank
        self.num_epochs = cfg.train.num_epochs
        self.num_classes = cfg.dataset.num_classes
        self.lambda_coarse = cfg.loss.lambda_coarse
        self.soft_beta = cfg.loss.soft_beta

        # --- NEW: extended loss options ---
        self.dual_temp_coarse = getattr(cfg.loss, 'dual_temp_coarse', 1.0)
        self.dual_temp_fine = getattr(cfg.loss, 'dual_temp_fine', 1.0)
        self.nc_reg_weight = getattr(cfg.loss, 'nc_reg_weight', 0.0)
        self.center_loss_weight = getattr(cfg.loss, 'center_loss_weight', 0.0)
        self.lambda_warmup = getattr(cfg.loss, 'lambda_warmup', False)
        self.lambda_decay = getattr(cfg.loss, 'lambda_decay', False)
        self.coarse_detach = getattr(cfg.loss, 'coarse_detach', False)
        self.two_stage_mode = getattr(cfg.loss, 'two_stage_mode', 'joint')

        # NC reg: running class means (updated each batch)
        if self.nc_reg_weight > 0:
            self._class_means = None
            self._class_counts = None

        self.init_all_params()

    def _is_hier(self):
        ht = self.cfg.dataset.hier_type
        if ht == 'default':
            return self.cfg.dataset.num_classes_1 > 0 and self.cfg.dataset.num_classes_2 > 0
        return ht != 'flat'

    def init_all_params(self):
        self.mixup_alpha = self.cfg.train.trainer.mixup_alpha

    def reset_epoch(self, epoch):
        self.epoch = epoch

    def forward(self, model, criterion, data, targets, **kwargs):
        return getattr(Trainer, self.type)(
            self, model, criterion, data, targets, **kwargs)

    def _with_autocast(self):
        return torch.cuda.amp.autocast() if self.cfg.mixed_precision else nullcontext()

    def _with_freeze(self):
        return torch.no_grad() if self.cfg.backbone.backbone_freeze else nullcontext()

    def _get_lambda(self):
        """Get current λ value. If warmup, anneals from 1.0 → lambda_coarse."""
        if self.two_stage_mode == 'coarse_only':
            return 1.0
        if self.two_stage_mode == 'fine_only':
            return 0.0
        if self.lambda_decay:
            progress = self.epoch / self.num_epochs
            return self.lambda_coarse * max(0, 1.0 - progress)
        if not self.lambda_warmup:
            return self.lambda_coarse

        # Linear annealing: first 25% epochs → λ=1.0, then linear to lambda_coarse
        warmup_end = self.num_epochs * 0.25
        if self.epoch <= warmup_end:
            return 1.0
        else:
            progress = (self.epoch - warmup_end) / (self.num_epochs - warmup_end)
            return 1.0 + (self.lambda_coarse - 1.0) * progress

    def _coarse_loss(self, coarse_logits, coarse_targets):
        """Coarse CE with optional soft label + dual temperature."""
        # Apply dual temperature
        scaled_logits = coarse_logits / self.dual_temp_coarse

        if self.soft_beta > 0:
            K = scaled_logits.size(1)
            soft = torch.full_like(scaled_logits, self.soft_beta / (K - 1))
            soft.scatter_(1, coarse_targets.unsqueeze(1), 1.0 - self.soft_beta)
            log_probs = F.log_softmax(scaled_logits, dim=1)
            return -(soft * log_probs).sum(dim=1).mean()
        return F.cross_entropy(scaled_logits, coarse_targets)

    def _fine_loss(self, fine_logits, fine_targets):
        """Fine CE with dual temperature."""
        scaled_logits = fine_logits / self.dual_temp_fine
        return F.cross_entropy(scaled_logits, fine_targets)

    def _nc_regularizer(self, features, targets):
        """NC-aware regularizer: push class means toward ETF geometry.

        L_nc = Σ_{i≠j} (cos(μ_i, μ_j) - (-1/(K-1)))²

        Uses running means updated with EMA for stability.
        """
        if self.nc_reg_weight <= 0:
            return torch.tensor(0.0, device=features.device)

        B, D = features.shape
        K = self.num_classes

        # Initialize running means
        if self._class_means is None:
            self._class_means = torch.zeros(K, D, device=features.device)
            self._class_counts = torch.zeros(K, device=features.device)

        # Update running means with current batch (EMA, momentum=0.1)
        with torch.no_grad():
            for c in range(K):
                mask = (targets == c)
                if mask.sum() > 0:
                    batch_mean = features[mask].mean(dim=0)
                    if self._class_counts[c] == 0:
                        self._class_means[c] = batch_mean
                    else:
                        self._class_means[c] = 0.9 * self._class_means[c] + 0.1 * batch_mean
                    self._class_counts[c] += mask.sum()

        # Compute NC2 loss on running means
        # Only use classes with enough samples
        valid = self._class_counts > 0
        if valid.sum() < 2:
            return torch.tensor(0.0, device=features.device)

        means = self._class_means[valid]  # [K', D]
        K_valid = means.shape[0]

        # Center
        global_mean = means.mean(dim=0)
        centered = means - global_mean

        # Normalize
        norms = centered.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normed = centered / norms

        # Cosine similarity matrix
        cos_sim = normed @ normed.T  # [K', K']

        # Target: -1/(K-1) for off-diagonal
        ideal = -1.0 / (K_valid - 1)

        # Loss: MSE of off-diagonal elements vs ideal
        mask = ~torch.eye(K_valid, dtype=torch.bool, device=features.device)
        loss = ((cos_sim[mask] - ideal) ** 2).mean()

        return loss

    def default(self, model, criterion, data, targets, tgts_1=None, tgts_2=None, **kwargs):
        data = data.cuda(self.rank)
        targets = targets.cuda(self.rank)
        if tgts_1 is not None:
            tgts_1 = tgts_1.cuda(self.rank)

        current_lambda = self._get_lambda()

        with self._with_autocast():
            with self._with_freeze():
                features = model(data, feature_flag=True)
            outputs = model(features, classifier_flag=True)

            if isinstance(outputs, tuple):
                fine_logits, coarse_logits = outputs
                if self.coarse_detach:
                    coarse_det = model(features.detach(), classifier_flag=True)[1]
                    loss = self._fine_loss(fine_logits, targets) \
                         + current_lambda * self._coarse_loss(coarse_det, tgts_1)
                else:
                    loss = self._fine_loss(fine_logits, targets) \
                         + current_lambda * self._coarse_loss(coarse_logits, tgts_1)
            else:
                fine_logits = outputs
                loss = self._fine_loss(fine_logits, targets)

            # NC regularizer
            if self.nc_reg_weight > 0:
                nc_loss = self._nc_regularizer(features.detach(), targets)
                loss = loss + self.nc_reg_weight * nc_loss

        pred = torch.argmax(fine_logits, 1)
        acc = accuracy(pred.cpu().numpy(), targets.cpu().numpy())[0]

        return loss, acc

    def mixup(self, model, criterion, data, targets, tgts_1=None, tgts_2=None, **kwargs):
        data = data.cuda(self.rank)
        targets = targets.cuda(self.rank)
        if tgts_1 is not None:
            tgts_1 = tgts_1.cuda(self.rank)

        current_lambda = self._get_lambda()

        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha) if self.mixup_alpha > 0 else 1
        index = torch.randperm(data.size(0)).cuda(self.rank)
        mixed_x = lam * data + (1 - lam) * data[index]

        with self._with_autocast():
            with self._with_freeze():
                mixed_features = model(mixed_x, feature_flag=True)
            outputs = model(mixed_features, classifier_flag=True)

            if isinstance(outputs, tuple):
                fine_logits, coarse_logits = outputs
                loss = lam * self._fine_loss(fine_logits, targets) \
                     + (1 - lam) * self._fine_loss(fine_logits, targets[index]) \
                     + current_lambda * (
                         lam * self._coarse_loss(coarse_logits, tgts_1) +
                         (1 - lam) * self._coarse_loss(coarse_logits, tgts_1[index]))
            else:
                loss = mixup_utils.mixup_criterion(
                    criterion, outputs, targets, targets[index], lam)

        with torch.no_grad():
            plain_outputs = model(data)
        if isinstance(plain_outputs, tuple):
            fine_logits = plain_outputs[0]
        else:
            fine_logits = plain_outputs
        pred = torch.argmax(fine_logits, 1)
        acc = accuracy(pred.cpu().numpy(), targets.cpu().numpy())[0]

        return loss, acc
