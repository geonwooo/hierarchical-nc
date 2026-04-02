import torch
import math
import numpy as np
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

    def _coarse_loss(self, coarse_logits, coarse_targets):
        """Coarse CE: soft label이면 soft, 아니면 hard."""
        if self.soft_beta > 0:
            K = coarse_logits.size(1)
            soft = torch.full_like(coarse_logits, self.soft_beta / (K - 1))
            soft.scatter_(1, coarse_targets.unsqueeze(1), 1.0 - self.soft_beta)
            log_probs = F.log_softmax(coarse_logits, dim=1)
            return -(soft * log_probs).sum(dim=1).mean()
        return F.cross_entropy(coarse_logits, coarse_targets)

    def default(self, model, criterion, data, targets, tgts_1=None, tgts_2=None, **kwargs):
        data = data.cuda(self.rank)
        targets = targets.cuda(self.rank)
        if tgts_1 is not None:
            tgts_1 = tgts_1.cuda(self.rank)

        with self._with_autocast():
            with self._with_freeze():
                features = model(data, feature_flag=True)
            outputs = model(features, classifier_flag=True)

            if isinstance(outputs, tuple):
                # B/C/D/E/F: (fine_100, coarse_20)
                fine_logits, coarse_logits = outputs
                loss = criterion(fine_logits, targets) \
                     + self.lambda_coarse * self._coarse_loss(coarse_logits, tgts_1)
            else:
                # A: flat 100-way
                fine_logits = outputs
                loss = criterion(fine_logits, targets)

        # 모든 모델 같은 metric: fine_100 accuracy
        pred = torch.argmax(fine_logits, 1)
        acc = accuracy(pred.cpu().numpy(), targets.cpu().numpy())[0]

        return loss, acc

    def mixup(self, model, criterion, data, targets, tgts_1=None, tgts_2=None, **kwargs):
        data = data.cuda(self.rank)
        targets = targets.cuda(self.rank)
        if tgts_1 is not None:
            tgts_1 = tgts_1.cuda(self.rank)

        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha) if self.mixup_alpha > 0 else 1
        index = torch.randperm(data.size(0)).cuda(self.rank)
        mixed_x = lam * data + (1 - lam) * data[index]

        with self._with_autocast():
            with self._with_freeze():
                mixed_features = model(mixed_x, feature_flag=True)
            outputs = model(mixed_features, classifier_flag=True)

            if isinstance(outputs, tuple):
                fine_logits, coarse_logits = outputs
                loss = mixup_utils.mixup_criterion(
                    criterion, fine_logits, targets, targets[index], lam) \
                     + self.lambda_coarse * (
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
