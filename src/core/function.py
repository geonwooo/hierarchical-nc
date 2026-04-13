import torch
import torch.nn.functional as F

import os
import json
import time
import numpy as np
from collections import defaultdict

import _init_paths
from core.evaluate import accuracy, AverageMeter
from utils.utils import get_model

TRUE_SEQ_TYPES = {
    'true_seq_direct', 'true_seq_residual',
    'true_seq_probw', 'true_seq_probw_nosg',
}


def _unpack_batch(batch):
    data, targets = batch[0], batch[1]
    tgts_1 = batch[2] if len(batch) > 2 else None
    tgts_2 = batch[3] if len(batch) > 3 else None
    return data, targets, tgts_1, tgts_2


def _is_true_seq(cfg):
    ht = cfg.dataset.hier_type
    if ht == 'default' and cfg.dataset.num_classes_1 > 0:
        return True
    return ht in TRUE_SEQ_TYPES


def _get_fine_logits(outputs, cfg):
    if isinstance(outputs, tuple):
        if _is_true_seq(cfg):
            return None
        return outputs[0]
    return outputs


def _build_local_to_fine(cfg):
    """(coarse_group, local_idx) -> fine_class lookup."""
    from modules.hier_ops import FINE_TO_COARSE

    grouping_file = getattr(cfg.dataset, 'grouping_file', 'none')
    if grouping_file and grouping_file != 'none':
        with open(grouping_file, 'r') as f:
            gdata = json.load(f)
        f2c = gdata['fine_to_coarse']
    elif cfg.dataset.random_hierarchy:
        rng = np.random.RandomState(cfg.seed_num)
        perm = rng.permutation(100)
        f2c = [0] * 100
        for g in range(20):
            for r in range(5):
                f2c[perm[g * 5 + r]] = g
    else:
        f2c = FINE_TO_COARSE

    groups = defaultdict(list)
    for c, g in enumerate(f2c):
        groups[g].append(c)

    local_to_fine = {}
    for g in groups:
        for li, c in enumerate(sorted(groups[g])):
            local_to_fine[(g, li)] = c

    return local_to_fine


def train_model(
    trainloader, model, epoch, num_epochs, optimizer, trainer,
    criterion, cfg, logger, verbose, **kwargs
):
    if cfg.eval_mode:
        model.eval()
    else:
        model.train()

    start_time = time.time()
    num_batches = len(trainloader)
    if 'num_batches' in kwargs:
        num_batches = kwargs['num_batches']
    scaler = kwargs.get('scaler', None)
    tr_loss = AverageMeter()
    tr_acc = AverageMeter()

    trainer.reset_epoch(epoch)
    for i, batch in enumerate(trainloader):
        if i > num_batches - 1:
            break
        data, targets, tgts_1, tgts_2 = _unpack_batch(batch)
        cnt = targets.shape[0]
        loss, acc = trainer.forward(
            model, criterion, data, targets,
            tgts_1=tgts_1, tgts_2=tgts_2)

        tr_loss.update(loss.data.item(), cnt)
        tr_acc.update(acc, cnt)

        if cfg.mixed_precision and scaler is not None:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (i % cfg.show_step == 0) and verbose:
            metric = "joint" if _is_true_seq(cfg) else "fine100"
            pbar_str = "Epoch:{:>3d} [{:>3d}/{}] loss:{:>5.3f} {}_acc:{:>5.2f}%".format(
                epoch, i, num_batches, tr_loss.val, metric, tr_acc.val * 100)
            logger.info(pbar_str)

    end_time = time.time()
    if verbose:
        metric = "joint" if _is_true_seq(cfg) else "fine100"
        pbar_str = "---Epoch:{:>3d}/{} tr_loss:{:>5.3f} {}_acc:{:>5.2f}% elapsed:{:>5.2f}m---".format(
            epoch, num_epochs, tr_loss.avg, metric, tr_acc.avg * 100,
            (end_time - start_time) / 60)
        logger.info(pbar_str)

    return tr_acc.avg, tr_loss.avg


def valid_model(
    dataloader, model, epoch,
    criterion, cfg, logger, verbose, rank, **kwargs
):
    model.eval()
    is_seq = _is_true_seq(cfg)

    with torch.no_grad():
        val_loss = AverageMeter()
        val_joint = AverageMeter()
        val_coarse = AverageMeter()
        val_fine_oracle = AverageMeter()
        val_fine100 = AverageMeter()

        local_to_fine = _build_local_to_fine(cfg) if is_seq else None

        for i, batch in enumerate(dataloader):
            data, targets, tgts_1, tgts_2 = _unpack_batch(batch)
            data = data.cuda(rank)
            targets_gpu = targets.cuda(rank)
            B = targets.shape[0]

            features = model(data, feature_flag=True)
            outputs = model(features, classifier_flag=True)

            if is_seq and isinstance(outputs, tuple):
                coarse_logits, fine_all = outputs  # (B,20), (B,20,5)
                tgts_1_gpu = tgts_1.cuda(rank)

                # Loss (coarse only for monitoring)
                loss = F.cross_entropy(coarse_logits, tgts_1_gpu)

                pred_c = torch.argmax(coarse_logits, 1)
                tgts_1_np = tgts_1.numpy()
                tgts_2_np = tgts_2.numpy()

                # Coarse acc
                pred_c_np = pred_c.cpu().numpy()
                coarse_acc = (pred_c_np == tgts_1_np).mean()

                # Fine oracle: GT group으로 fine 선택
                fine_oracle_logits = fine_all[
                    torch.arange(B, device=data.device), tgts_1_gpu]
                pred_f_oracle = torch.argmax(
                    fine_oracle_logits, 1).cpu().numpy()
                fine_oracle_acc = (pred_f_oracle == tgts_2_np).mean()

                # Joint: predicted group으로 fine 선택
                fine_pred_logits = fine_all[
                    torch.arange(B, device=data.device), pred_c]
                pred_f = torch.argmax(fine_pred_logits, 1).cpu().numpy()
                joint_acc = ((pred_c_np == tgts_1_np) &
                             (pred_f == tgts_2_np)).mean()

                # 100-way 복원
                pred_100 = np.array([
                    local_to_fine.get(
                        (int(pred_c_np[j]), int(pred_f[j])), -1)
                    for j in range(B)])
                fine100_acc = (pred_100 == targets.numpy()).mean()

                val_loss.update(loss.data.item(), B)
                val_joint.update(joint_acc, B)
                val_coarse.update(coarse_acc, B)
                val_fine_oracle.update(fine_oracle_acc, B)
                val_fine100.update(fine100_acc, B)

            else:
                fine_logits = _get_fine_logits(outputs, cfg)
                loss = criterion(fine_logits, targets_gpu)
                pred = torch.argmax(fine_logits, 1)
                acc, cnt = accuracy(
                    pred.cpu().numpy(), targets.numpy())
                val_loss.update(loss.data.item(), B)
                val_joint.update(acc, B)

        if cfg.ddp:
            val_loss.all_reduce()
            val_joint.all_reduce()

    if verbose:
        if is_seq:
            pbar_str = (
                "------Valid: Epoch:{:>3d}"
                " loss:{:>5.3f}"
                " joint:{:>5.2f}%"
                " coarse:{:>5.2f}%"
                " fine_oracle:{:>5.2f}%"
                " 100way:{:>5.2f}%".format(
                    epoch, val_loss.avg,
                    val_joint.avg * 100,
                    val_coarse.avg * 100,
                    val_fine_oracle.avg * 100,
                    val_fine100.avg * 100))
        else:
            pbar_str = "------Valid: Epoch:{:>3d} val_loss:{:>5.3f} fine_acc:{:>5.2f}%".format(
                epoch, val_loss.avg, val_joint.avg * 100)
        logger.info(pbar_str)

    return val_joint.avg, val_loss.avg


def test_model(
    dataloader, cfg, rank, verbose,
    num_classes=10, pretrained=None
):
    model = get_model(cfg, num_classes, rank)
    is_seq = _is_true_seq(cfg)

    if os.path.isfile(pretrained):
        print("=> loading checkpoint '{}'".format(pretrained))
        checkpoint = torch.load(
            pretrained, map_location='cuda:{}'.format(rank))
        if cfg.ddp or cfg.dp:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            ckpt_state_dict = dict()
            for k, v in checkpoint['state_dict'].items():
                if k.startswith('module'):
                    ckpt_state_dict[k[7:]] = v
                else:
                    ckpt_state_dict[k] = v
            model.load_state_dict(ckpt_state_dict)
    model.eval()

    local_to_fine = _build_local_to_fine(cfg) if is_seq else None

    with torch.no_grad():
        ts_joint = AverageMeter()
        ts_coarse = AverageMeter()
        ts_fine100 = AverageMeter()

        for i, batch in enumerate(dataloader):
            data, targets, tgts_1, tgts_2 = _unpack_batch(batch)
            data = data.cuda(rank)
            B = targets.shape[0]

            outputs = model(data)

            if is_seq and isinstance(outputs, tuple):
                coarse_logits, fine_all = outputs
                pred_c = torch.argmax(coarse_logits, 1)
                pred_c_np = pred_c.cpu().numpy()

                fine_pred_logits = fine_all[
                    torch.arange(B, device=data.device), pred_c]
                pred_f = torch.argmax(fine_pred_logits, 1).cpu().numpy()

                joint = ((pred_c_np == tgts_1.numpy()) &
                         (pred_f == tgts_2.numpy())).mean()
                coarse = (pred_c_np == tgts_1.numpy()).mean()

                pred_100 = np.array([
                    local_to_fine.get(
                        (int(pred_c_np[j]), int(pred_f[j])), -1)
                    for j in range(B)])
                fine100 = (pred_100 == targets.numpy()).mean()

                ts_joint.update(joint, B)
                ts_coarse.update(coarse, B)
                ts_fine100.update(fine100, B)
            else:
                fine_logits = _get_fine_logits(outputs, cfg)
                pred = torch.argmax(fine_logits, 1)
                acc, cnt = accuracy(pred.cpu().numpy(), targets.numpy())
                ts_joint.update(acc, B)

        if cfg.ddp:
            ts_joint.all_reduce()

    if verbose:
        if is_seq:
            print("*** Test Joint:{:>5.2f}% Coarse:{:>5.2f}% 100way:{:>5.2f}%".format(
                ts_joint.avg * 100, ts_coarse.avg * 100, ts_fine100.avg * 100))
        else:
            print("*** Test fine_100 Accuracy: {:>5.2f}%".format(
                ts_joint.avg * 100))


def collect_nc_stats_inline(model, dataloader, num_classes, rank,
                            hier_type='flat', f2c=None):
    mm = model.module if hasattr(model, 'module') else model
    mm.eval()

    class_features = defaultdict(list)
    with torch.no_grad():
        for batch in dataloader:
            data = batch[0].cuda(rank)
            targets = batch[1]
            features = mm.extract_feature(data)
            for i in range(features.size(0)):
                class_features[targets[i].item()].append(
                    features[i].cpu().numpy())

    D = len(class_features[0][0])

    class_means = np.zeros((num_classes, D))
    for c in range(num_classes):
        if len(class_features[c]) > 0:
            class_means[c] = np.mean(class_features[c], axis=0)

    fine_nc1 = _compute_nc1(class_means, class_features, num_classes, D)
    result = {'fine_nc1': fine_nc1}

    if f2c is not None:
        num_groups = len(set(f2c))
        group_features = defaultdict(list)
        for c in range(num_classes):
            group_features[f2c[c]].extend(class_features[c])

        group_means = np.zeros((num_groups, D))
        for g in range(num_groups):
            if len(group_features[g]) > 0:
                group_means[g] = np.mean(group_features[g], axis=0)

        result['coarse_nc1'] = _compute_nc1(
            group_means, group_features, num_groups, D)

    mm.train()
    return result


def _compute_nc1(class_means, class_features, K, D):
    global_mean = class_means.mean(axis=0)

    Sw = np.zeros((D, D))
    count = 0
    for c in range(K):
        if c not in class_features or len(class_features[c]) == 0:
            continue
        feats = np.array(class_features[c])
        centered = feats - class_means[c]
        Sw += centered.T @ centered
        count += len(feats)
    Sw /= max(count, 1)

    centered_means = class_means - global_mean
    Sb = centered_means.T @ centered_means / K

    try:
        Sb_inv = np.linalg.pinv(Sb)
        nc1 = np.trace(Sw @ Sb_inv) / K
    except Exception:
        nc1 = float('inf')

    return float(nc1)
