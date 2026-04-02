import torch
import torch.nn.functional as F

import os
import time
import numpy as np

import _init_paths
from core.evaluate import accuracy, AverageMeter, FusionMatrix
from utils.utils import get_model


def _unpack_batch(batch):
    """Unpack a dataloader batch into (data, targets, tgts_1, tgts_2).
    tgts_1 and tgts_2 are None for non-hierarchical datasets.
    """
    data, targets = batch[0], batch[1]
    tgts_1 = batch[2] if len(batch) > 2 else None
    tgts_2 = batch[3] if len(batch) > 3 else None
    return data, targets, tgts_1, tgts_2


def _hier_acc(output_1, output_2, tgts_1, tgts_2):
    """Joint accuracy: correct only when both hierarchical predictions match."""
    pred_1 = torch.argmax(output_1, 1).cpu().numpy()
    pred_2 = torch.argmax(output_2, 1).cpu().numpy()
    correct = (pred_1 == tgts_1.cpu().numpy()) & (pred_2 == tgts_2.cpu().numpy())
    return correct.mean(), len(correct)


def _resolve_hier_type(cfg):
    ht = cfg.dataset.hier_type
    if ht == 'default':
        if cfg.dataset.num_classes_1 > 0 and cfg.dataset.num_classes_2 > 0:
            return 'sequential'
        return 'flat'
    return ht


def train_model(
    trainloader, model, epoch, num_epochs, optimizer, trainer,
    criterion, cfg, logger, verbose, **kwargs
):
    if cfg.eval_mode:
        model.eval()
    else:
        if cfg.backbone.backbone_freeze:
            mm = model.module if cfg.ddp or cfg.dp else model
            mm.backbone.eval()
            mm.pooling.eval()
            mm.reshape.eval()
            # classifier might have different names now, just train all remaining
            model.train()
            if hasattr(mm, 'backbone'):
                mm.backbone.eval()
        else:
            model.train()

    start_time = time.time()

    num_batches = len(trainloader) if 'num_batches' not in kwargs else kwargs['num_batches']
    scaler = None if 'scaler' not in kwargs else kwargs['scaler']

    tr_loss = AverageMeter()
    tr_acc = AverageMeter()

    trainer.reset_epoch(epoch)
    for i, batch in enumerate(trainloader):
        if i > num_batches - 1:
            break
        data, targets, tgts_1, tgts_2 = _unpack_batch(batch)
        cnt = targets.shape[0]
        loss, acc = trainer.forward(
            model, criterion, data, targets, tgts_1=tgts_1, tgts_2=tgts_2)

        tr_loss.update(loss.data.item(), cnt)
        tr_acc.update(acc, cnt)

        if cfg.mixed_precision:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (i & cfg.show_step == 0) and verbose:
            pbar_str = "Epoch:{:>3d} [{:>3d}/{}] loss:{:>5.3f} acc:{:>5.2f}%".format(
                epoch, i, num_batches, tr_loss.val, tr_acc.val * 100)
            logger.info(pbar_str)
    end_time = time.time()
    if verbose:
        pbar_str = "---Epoch:{:>3d}/{}".format(epoch, num_epochs) \
            + " tr_loss:{:>5.3f}".format(tr_loss.avg) \
            + " tr_acc:{:>5.2f}%".format(tr_acc.avg * 100) \
            + " elapsed_time:{:>5.2f}m---".format((end_time - start_time)/60)
        logger.info(pbar_str)

    return tr_acc.avg, tr_loss.avg


def valid_model(
    dataloader, model, epoch,
    criterion, cfg, logger, verbose, rank, **kwargs
):
    model.eval()
    ht = _resolve_hier_type(cfg)

    with torch.no_grad():
        val_loss = AverageMeter()
        val_acc = AverageMeter()
        val_fine_acc = AverageMeter()     # NEW: always track fine acc
        val_coarse_acc = AverageMeter()   # NEW: always track coarse acc

        for i, batch in enumerate(dataloader):
            data, targets, tgts_1, tgts_2 = _unpack_batch(batch)
            data = data.cuda(rank)
            targets = targets.cuda(rank)
            if tgts_1 is not None:
                tgts_1 = tgts_1.cuda(rank)
            if tgts_2 is not None:
                tgts_2 = tgts_2.cuda(rank)

            features = model(data, feature_flag=True)
            outputs = model(features, classifier_flag=True)

            if ht == 'factorized':
                fine_logits, coarse_logits = outputs
                loss = criterion(fine_logits, targets) \
                     + cfg.loss.lambda_coarse * criterion(coarse_logits, tgts_1)
                # Fine accuracy (primary)
                pred_fine = torch.argmax(fine_logits, 1)
                acc, cnt = accuracy(pred_fine.cpu().numpy(), targets.cpu().numpy())
                val_fine_acc.update(acc, cnt)
                # Coarse accuracy (secondary)
                pred_coarse = torch.argmax(coarse_logits, 1)
                c_acc, _ = accuracy(pred_coarse.cpu().numpy(), tgts_1.cpu().numpy())
                val_coarse_acc.update(c_acc, cnt)

            elif ht in ('sequential', 'sequential_residual'):
                output_1, output_2 = outputs
                loss = criterion(output_1, tgts_1) + criterion(output_2, tgts_2)
                acc, cnt = _hier_acc(output_1, output_2, tgts_1, tgts_2)
                # Also report individual level accuracies
                pred_1 = torch.argmax(output_1, 1)
                c_acc, _ = accuracy(pred_1.cpu().numpy(), tgts_1.cpu().numpy())
                val_coarse_acc.update(c_acc, cnt)

            else:  # flat
                loss = criterion(outputs, targets)
                pred = torch.argmax(outputs, 1)
                acc, cnt = accuracy(pred.cpu().numpy(), targets.cpu().numpy())
                val_fine_acc.update(acc, cnt)

            val_loss.update(loss.data.item(), targets.shape[0])
            val_acc.update(acc, cnt)

        if cfg.ddp:
            val_loss.all_reduce()
            val_acc.all_reduce()
            val_fine_acc.all_reduce()
            val_coarse_acc.all_reduce()

    if verbose:
        pbar_str = "------Valid: Epoch:{:>3d}".format(epoch) \
            + " val_loss:{:>5.3f}".format(val_loss.avg) \
            + " val_acc:{:>5.2f}%".format(val_acc.avg * 100)
        if val_fine_acc.count > 0:
            pbar_str += " fine:{:>5.2f}%".format(val_fine_acc.avg * 100)
        if val_coarse_acc.count > 0:
            pbar_str += " coarse:{:>5.2f}%".format(val_coarse_acc.avg * 100)
        logger.info(pbar_str)

    return val_acc.avg, val_loss.avg


def test_model(
    dataloader, cfg, rank, verbose,
    num_classes=10, pretrained=None
):
    model = get_model(cfg, num_classes, rank)
    ht = _resolve_hier_type(cfg)

    if os.path.isfile(pretrained):
        print("=> loading checkpoint '{}'".format(pretrained))
        checkpoint = torch.load(pretrained, map_location='cuda:{}'.format(rank))
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

    with torch.no_grad():
        ts_acc = AverageMeter()
        ts_fine_acc = AverageMeter()
        ts_coarse_acc = AverageMeter()

        for i, batch in enumerate(dataloader):
            data, targets, tgts_1, tgts_2 = _unpack_batch(batch)
            data = data.cuda(rank)

            outputs = model(data)

            if ht == 'factorized':
                fine_logits, coarse_logits = outputs
                pred_fine = torch.argmax(fine_logits, 1)
                acc, cnt = accuracy(pred_fine.cpu().numpy(), targets.numpy())
                ts_fine_acc.update(acc, cnt)
                pred_coarse = torch.argmax(coarse_logits, 1)
                c_acc, _ = accuracy(pred_coarse.cpu().numpy(), tgts_1.numpy())
                ts_coarse_acc.update(c_acc, cnt)

            elif ht in ('sequential', 'sequential_residual'):
                output_1, output_2 = outputs
                acc, cnt = _hier_acc(
                    output_1, output_2,
                    tgts_1.cuda(rank), tgts_2.cuda(rank))
                pred_1 = torch.argmax(output_1, 1)
                c_acc, _ = accuracy(pred_1.cpu().numpy(), tgts_1.numpy())
                ts_coarse_acc.update(c_acc, cnt)

            else:
                pred = torch.argmax(outputs, 1)
                acc, cnt = accuracy(pred.cpu().numpy(), targets.numpy())

            ts_acc.update(acc, cnt)

        if cfg.ddp:
            ts_acc.all_reduce()
            ts_fine_acc.all_reduce()
            ts_coarse_acc.all_reduce()

    if verbose:
        msg = "*** Test Accuracy: {:>5.2f}%".format(ts_acc.avg * 100)
        if ts_fine_acc.count > 0:
            msg += "  (fine_100: {:>5.2f}%)".format(ts_fine_acc.avg * 100)
        if ts_coarse_acc.count > 0:
            msg += "  (coarse_20: {:>5.2f}%)".format(ts_coarse_acc.avg * 100)
        print(msg)
