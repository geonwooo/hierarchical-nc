import torch
import math
import numpy as np
from collections import defaultdict
from contextlib import nullcontext
import torch.nn.functional as F

import utils.mixup_utils as mixup_utils
from core.evaluate import accuracy

TRUE_SEQ_TYPES = {
    'true_seq_direct', 'true_seq_residual',
    'true_seq_probw', 'true_seq_probw_nosg',
}


class Trainer:
    def __init__(self, cfg, rank):
        self.cfg = cfg
        self.type = cfg.train.trainer.type
        self.rank = rank
        self.num_epochs = cfg.train.num_epochs
        self.num_classes = cfg.dataset.num_classes
        self.lambda_coarse = cfg.loss.lambda_coarse
        self.soft_beta = getattr(cfg.loss, 'soft_beta', 0.0)

        self.hier_type = cfg.dataset.hier_type
        if self.hier_type == 'default':
            if cfg.dataset.num_classes_1 > 0 and cfg.dataset.num_classes_2 > 0:
                self.hier_type = 'true_seq_direct'
            else:
                self.hier_type = 'flat'
        self.is_true_sequential = (self.hier_type in TRUE_SEQ_TYPES)

        self.dual_temp_coarse = getattr(cfg.loss, 'dual_temp_coarse', 1.0)
        self.dual_temp_fine = getattr(cfg.loss, 'dual_temp_fine', 1.0)
        self.nc_reg_weight = getattr(cfg.loss, 'nc_reg_weight', 0.0)
        self.lambda_warmup = getattr(cfg.loss, 'lambda_warmup', False)
        self.lambda_decay = getattr(cfg.loss, 'lambda_decay', False)
        self.coarse_detach = getattr(cfg.loss, 'coarse_detach', False)
        self.two_stage_mode = getattr(cfg.loss, 'two_stage_mode', 'joint')

        self.scheduled_sampling = getattr(cfg.loss, 'scheduled_sampling', False)
        self.aux100_weight = getattr(cfg.loss, 'aux100_weight', 0.0)
        self.fine_label_smooth = getattr(cfg.loss, 'fine_label_smooth', 0.0)
        self.joint_100way = getattr(cfg.loss, 'joint_100way', False)

        self.init_all_params()

    def init_all_params(self):
        self.mixup_alpha = self.cfg.train.trainer.mixup_alpha

    def reset_epoch(self, epoch):
        self.epoch = epoch

    def forward(self, model, criterion, data, targets, **kwargs):
        return getattr(Trainer, self.type)(
            self, model, criterion, data, targets, **kwargs)

    def _with_autocast(self):
        return (torch.cuda.amp.autocast()
                if self.cfg.mixed_precision else nullcontext())

    def _with_freeze(self):
        return (torch.no_grad()
                if self.cfg.backbone.backbone_freeze else nullcontext())

    def _get_lambda(self):
        if self.two_stage_mode == 'coarse_only':
            return 1.0
        if self.two_stage_mode == 'fine_only':
            return 0.0
        if self.lambda_decay:
            progress = self.epoch / self.num_epochs
            return self.lambda_coarse * max(0, 1.0 - progress)
        if not self.lambda_warmup:
            return self.lambda_coarse
        warmup_end = self.num_epochs * 0.25
        if self.epoch <= warmup_end:
            return 1.0
        progress = (self.epoch - warmup_end) / (self.num_epochs - warmup_end)
        return 1.0 + (self.lambda_coarse - 1.0) * progress

    def _coarse_loss(self, coarse_logits, coarse_targets):
        scaled = coarse_logits / self.dual_temp_coarse
        if self.soft_beta > 0:
            K = scaled.size(1)
            soft = torch.full_like(scaled, self.soft_beta / (K - 1))
            soft.scatter_(1, coarse_targets.unsqueeze(1), 1.0 - self.soft_beta)
            log_probs = F.log_softmax(scaled, dim=1)
            return -(soft * log_probs).sum(dim=1).mean()
        return F.cross_entropy(scaled, coarse_targets)

    def _fine_loss(self, fine_logits, fine_targets):
        scaled = fine_logits / self.dual_temp_fine
        if self.fine_label_smooth > 0:
            K = scaled.size(1)
            soft = torch.full_like(scaled, self.fine_label_smooth / (K - 1))
            soft.scatter_(1, fine_targets.unsqueeze(1), 1.0 - self.fine_label_smooth)
            log_probs = F.log_softmax(scaled, dim=1)
            return -(soft * log_probs).sum(dim=1).mean()
        return F.cross_entropy(scaled, fine_targets)

    def _get_sampling_epsilon(self):
        if not self.scheduled_sampling:
            return 0.0
        return min(1.0, self.epoch / (self.num_epochs * 0.7))

    def _seq_joint_acc(self, coarse_logits, fine_all, tgts_1, tgts_2):
        B = coarse_logits.size(0)
        pred_c = torch.argmax(coarse_logits, 1)
        fine_logits = fine_all[torch.arange(B, device=pred_c.device), pred_c]
        pred_f = torch.argmax(fine_logits, 1)
        correct = ((pred_c == tgts_1) & (pred_f == tgts_2))
        return correct.float().mean().item()

    def _soft_routing_acc(self, coarse_logits, fine_all, targets, model):
        mm = model.module if hasattr(model, 'module') else model
        logit_100 = mm.assemble_100way(coarse_logits, fine_all)
        coarse_prob = F.softmax(coarse_logits, dim=1)
        fine_prob = F.softmax(fine_all, dim=2)
        class_prob = coarse_prob[:, mm.coarse_idx] * fine_prob[:, mm.coarse_idx, mm.local_idx]
        pred = torch.argmax(class_prob, dim=1)
        return (pred == targets).float().mean().item()

    def default(self, model, criterion, data, targets,
                tgts_1=None, tgts_2=None, **kwargs):
        data = data.cuda(self.rank)
        targets = targets.cuda(self.rank)
        if tgts_1 is not None:
            tgts_1 = tgts_1.cuda(self.rank)
        if tgts_2 is not None:
            tgts_2 = tgts_2.cuda(self.rank)

        current_lambda = self._get_lambda()

        with self._with_autocast():
            with self._with_freeze():
                features = model(data, feature_flag=True)
            outputs = model(features, classifier_flag=True)

            if self.is_true_sequential and isinstance(outputs, tuple):
                coarse_logits, fine_all = outputs
                B = coarse_logits.size(0)
                mm = model.module if hasattr(model, 'module') else model

                if self.joint_100way:
                    # Joint 100-way loss only
                    logit_100 = mm.assemble_100way(coarse_logits, fine_all)
                    loss = F.cross_entropy(logit_100, targets)
                else:
                    # Per-level loss
                    epsilon = self._get_sampling_epsilon()
                    if epsilon > 0 and np.random.random() < epsilon:
                        group_idx = torch.argmax(coarse_logits.detach(), 1)
                        coarse_mask = (group_idx == tgts_1)
                        fine_logits = fine_all[torch.arange(B, device=data.device), group_idx]
                        if coarse_mask.sum() > 0:
                            fine_loss = self._fine_loss(
                                fine_logits[coarse_mask], tgts_2[coarse_mask])
                        else:
                            fine_loss = torch.tensor(0.0, device=data.device)
                    else:
                        fine_logits = fine_all[torch.arange(B, device=data.device), tgts_1]
                        fine_loss = self._fine_loss(fine_logits, tgts_2)

                    coarse_loss = self._coarse_loss(coarse_logits, tgts_1)

                    if self.two_stage_mode == 'coarse_only':
                        loss = coarse_loss
                    elif self.two_stage_mode == 'fine_only':
                        loss = fine_loss
                    else:
                        loss = coarse_loss + current_lambda * fine_loss

                    if self.aux100_weight > 0 and hasattr(mm, 'assemble_100way'):
                        logit_100 = mm.assemble_100way(coarse_logits, fine_all)
                        loss = loss + self.aux100_weight * F.cross_entropy(logit_100, targets)

                acc = self._seq_joint_acc(
                    coarse_logits, fine_all, tgts_1, tgts_2)

            elif isinstance(outputs, tuple):
                fine_logits, coarse_logits = outputs
                loss = (self._fine_loss(fine_logits, targets)
                        + current_lambda
                        * self._coarse_loss(coarse_logits, tgts_1))
                pred = torch.argmax(fine_logits, 1)
                acc = accuracy(
                    pred.cpu().numpy(), targets.cpu().numpy())[0]
            else:
                fine_logits = outputs
                loss = self._fine_loss(fine_logits, targets)
                pred = torch.argmax(fine_logits, 1)
                acc = accuracy(
                    pred.cpu().numpy(), targets.cpu().numpy())[0]

        return loss, acc

    def mixup(self, model, criterion, data, targets,
              tgts_1=None, tgts_2=None, **kwargs):
        data = data.cuda(self.rank)
        targets = targets.cuda(self.rank)
        if tgts_1 is not None:
            tgts_1 = tgts_1.cuda(self.rank)
        if tgts_2 is not None:
            tgts_2 = tgts_2.cuda(self.rank)

        current_lambda = self._get_lambda()
        lam = (np.random.beta(self.mixup_alpha, self.mixup_alpha)
               if self.mixup_alpha > 0 else 1)
        index = torch.randperm(data.size(0)).cuda(self.rank)
        mixed_x = lam * data + (1 - lam) * data[index]

        with self._with_autocast():
            with self._with_freeze():
                mixed_features = model(mixed_x, feature_flag=True)
            outputs = model(mixed_features, classifier_flag=True)

            if self.is_true_sequential and isinstance(outputs, tuple):
                coarse_logits, fine_all = outputs
                B = coarse_logits.size(0)
                mm = model.module if hasattr(model, 'module') else model

                if self.joint_100way:
                    logit_100 = mm.assemble_100way(coarse_logits, fine_all)
                    loss = (lam * F.cross_entropy(logit_100, targets)
                            + (1 - lam) * F.cross_entropy(logit_100, targets[index]))
                else:
                    fine_a = fine_all[torch.arange(B, device=data.device), tgts_1]
                    fine_b = fine_all[torch.arange(B, device=data.device), tgts_1[index]]
                    loss = (lam * self._coarse_loss(coarse_logits, tgts_1)
                            + (1 - lam) * self._coarse_loss(coarse_logits, tgts_1[index])
                            + current_lambda * (
                                lam * self._fine_loss(fine_a, tgts_2)
                                + (1 - lam) * self._fine_loss(fine_b, tgts_2[index])))
            elif isinstance(outputs, tuple):
                fine_logits, coarse_logits = outputs
                loss = (lam * self._fine_loss(fine_logits, targets)
                        + (1 - lam) * self._fine_loss(fine_logits, targets[index])
                        + current_lambda * (
                            lam * self._coarse_loss(coarse_logits, tgts_1)
                            + (1 - lam) * self._coarse_loss(coarse_logits, tgts_1[index])))
            else:
                loss = mixup_utils.mixup_criterion(
                    criterion, outputs, targets, targets[index], lam)

        with torch.no_grad():
            plain_outputs = model(data)
        if self.is_true_sequential and isinstance(plain_outputs, tuple):
            acc = self._seq_joint_acc(plain_outputs[0], plain_outputs[1], tgts_1, tgts_2)
        elif isinstance(plain_outputs, tuple):
            pred = torch.argmax(plain_outputs[0], 1)
            acc = accuracy(pred.cpu().numpy(), targets.cpu().numpy())[0]
        else:
            pred = torch.argmax(plain_outputs, 1)
            acc = accuracy(pred.cpu().numpy(), targets.cpu().numpy())[0]

        return loss, acc
