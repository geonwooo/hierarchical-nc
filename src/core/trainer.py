import torch

import math
import numpy as np
from contextlib import nullcontext

import utils.mixup_utils as mixup_utils
from core.evaluate import accuracy


class Trainer:
    def __init__(self, cfg, rank):
        self.cfg = cfg
        self.type = cfg.train.trainer.type
        self.rank = rank
        self.num_epochs = cfg.train.num_epochs
        self.num_classes = cfg.dataset.num_classes
        self.init_all_params()

    def init_all_params(self):
        self.mixup_alpha = self.cfg.train.trainer.mixup_alpha

    def reset_epoch(self, epoch):
        self.epoch = epoch

    def forward(self, model, criterion, data, targets, **kwargs):
        return getattr(Trainer, self.type)(
            self, model, criterion, data, targets, **kwargs
        )

    def _with_autocast(self):
        return torch.cuda.amp.autocast() if self.cfg.mixed_precision else nullcontext()

    def _with_freeze(self):
        return torch.no_grad() if self.cfg.backbone.backbone_freeze else nullcontext()

    @staticmethod
    def _hier_acc(output_1, output_2, tgts_1, tgts_2):
        """Joint accuracy: correct only when both hierarchical predictions match."""
        pred_1 = torch.argmax(output_1, 1).cpu().numpy()
        pred_2 = torch.argmax(output_2, 1).cpu().numpy()
        correct = (pred_1 == tgts_1.cpu().numpy()) & (pred_2 == tgts_2.cpu().numpy())
        return correct.mean()

    def default(self, model, criterion, data, targets, tgts_1=None, tgts_2=None, **kwargs):
        data = data.cuda(self.rank)
        targets = targets.cuda(self.rank)
        if tgts_1 is not None:
            tgts_1 = tgts_1.cuda(self.rank)
        if tgts_2 is not None:
            tgts_2 = tgts_2.cuda(self.rank)

        with self._with_autocast():
            with self._with_freeze():
                features = model(data, feature_flag=True)
            outputs = model(features, classifier_flag=True)

            if isinstance(outputs, tuple):
                output_1, output_2 = outputs
                loss = criterion(output_1, tgts_1) + criterion(output_2, tgts_2)
            else:
                loss = criterion(outputs, targets)

        if isinstance(outputs, tuple):
            acc = self._hier_acc(output_1, output_2, tgts_1, tgts_2)
        else:
            pred = torch.argmax(outputs, 1)
            acc = accuracy(pred.cpu().numpy(), targets.cpu().numpy())[0]

        return loss, acc

    def mixup(self, model, criterion, data, targets, tgts_1=None, tgts_2=None, **kwargs):
        data = data.cuda(self.rank)
        targets = targets.cuda(self.rank)
        if tgts_1 is not None:
            tgts_1 = tgts_1.cuda(self.rank)
        if tgts_2 is not None:
            tgts_2 = tgts_2.cuda(self.rank)

        # Single permutation applied consistently across all label levels.
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha) if self.mixup_alpha > 0 else 1
        index = torch.randperm(data.size(0)).cuda(self.rank)
        mixed_x = lam * data + (1 - lam) * data[index]

        with self._with_autocast():
            with self._with_freeze():
                mixed_features = model(mixed_x, feature_flag=True)
            outputs = model(mixed_features, classifier_flag=True)

            if isinstance(outputs, tuple):
                output_1, output_2 = outputs
                loss = mixup_utils.mixup_criterion(
                    criterion, output_1, tgts_1, tgts_1[index], lam) \
                     + mixup_utils.mixup_criterion(
                    criterion, output_2, tgts_2, tgts_2[index], lam)
            else:
                loss = mixup_utils.mixup_criterion(
                    criterion, outputs, targets, targets[index], lam)

        with torch.no_grad():
            plain_outputs = model(data)

        if isinstance(plain_outputs, tuple):
            plain_1, plain_2 = plain_outputs
            acc = self._hier_acc(plain_1, plain_2, tgts_1, tgts_2)
        else:
            pred = torch.argmax(plain_outputs, 1)
            acc = accuracy(pred.cpu().numpy(), targets.cpu().numpy())[0]

        return loss, acc
