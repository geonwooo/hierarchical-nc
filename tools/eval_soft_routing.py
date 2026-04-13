#!/usr/bin/env python
"""
eval_soft_routing.py — 기존 모델 전부에 soft routing 적용.
Hard routing (argmax) vs Soft routing (확률 곱) 비교.

Usage: python tools/eval_soft_routing.py --rank 0
"""
import argparse
import glob
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'main'))
import _init_paths
from config import cfg, update_config
from utils.utils import get_model
import dataset as custom_dataset
from data_transform.transform_wrapper import get_transform
from torch.utils.data import DataLoader
from modules.hier_ops import FINE_TO_COARSE


def build_local_to_fine(f2c):
    groups = defaultdict(list)
    for c, g in enumerate(f2c):
        groups[g].append(c)
    local_to_fine = {}
    for g in groups:
        for li, c in enumerate(sorted(groups[g])):
            local_to_fine[(g, li)] = c
    return local_to_fine


def eval_model(model, dataloader, local_to_fine, num_groups, rank):
    model.eval()
    hard_correct = 0
    soft_correct = 0
    coarse_correct = 0
    fine_oracle_correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            data = batch[0].cuda(rank)
            targets = batch[1].numpy()
            tgts_1 = batch[2].numpy()
            tgts_2 = batch[3].numpy()
            B = len(targets)

            outputs = model(data)
            if not isinstance(outputs, tuple):
                continue

            coarse_logits, fine_all = outputs

            # Coarse accuracy
            pred_c = torch.argmax(coarse_logits, 1).cpu().numpy()
            coarse_correct += (pred_c == tgts_1).sum()

            # Fine oracle accuracy
            fine_oracle_logits = fine_all[
                torch.arange(B, device=data.device),
                torch.tensor(tgts_1, device=data.device)]
            pred_f_oracle = torch.argmax(fine_oracle_logits, 1).cpu().numpy()
            fine_oracle_correct += (pred_f_oracle == tgts_2).sum()

            # === Hard routing (기존) ===
            fine_hard = fine_all[
                torch.arange(B, device=data.device), 
                torch.argmax(coarse_logits, 1)]
            pred_f_hard = torch.argmax(fine_hard, 1).cpu().numpy()
            pred_100_hard = np.array([
                local_to_fine.get((int(pred_c[j]), int(pred_f_hard[j])), -1)
                for j in range(B)])
            hard_correct += (pred_100_hard == targets).sum()

            # === Soft routing (제안) ===
            coarse_prob = F.softmax(coarse_logits, dim=1)       # (B, 20)
            fine_prob = F.softmax(fine_all, dim=2)               # (B, 20, 5)
            # P(class c) = P(group g(c)) * P(c | group g(c))
            # 100개 class 전부의 확률 계산
            for j in range(B):
                best_prob = -1
                best_class = -1
                for g in range(num_groups):
                    p_g = coarse_prob[j, g].item()
                    for l in range(fine_prob.shape[2]):
                        c = local_to_fine.get((g, l), -1)
                        if c == -1:
                            continue
                        p_cl = p_g * fine_prob[j, g, l].item()
                        if p_cl > best_prob:
                            best_prob = p_cl
                            best_class = c
                if best_class == targets[j]:
                    soft_correct += 1

            total += B

    return {
        'hard_acc': hard_correct / total,
        'soft_acc': soft_correct / total,
        'coarse_acc': coarse_correct / total,
        'fine_oracle_acc': fine_oracle_correct / total,
        'total': total,
    }


def eval_model_fast(model, dataloader, f2c, num_groups, max_fpg, rank):
    """Vectorized soft routing — much faster."""
    model.eval()

    # Build index tensors
    num_classes = len(f2c)
    coarse_idx = torch.zeros(num_classes, dtype=torch.long, device='cuda:{}'.format(rank))
    local_idx = torch.zeros(num_classes, dtype=torch.long, device='cuda:{}'.format(rank))
    groups = defaultdict(list)
    for c, g in enumerate(f2c):
        groups[g].append(c)
    for g in groups:
        for li, c in enumerate(sorted(groups[g])):
            coarse_idx[c] = g
            local_idx[c] = li

    hard_correct = 0
    soft_correct = 0
    coarse_correct = 0
    fine_oracle_correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            data = batch[0].cuda(rank)
            targets = batch[1].cuda(rank)
            tgts_1 = batch[2].cuda(rank)
            tgts_2 = batch[3].cuda(rank)
            B = targets.size(0)

            outputs = model(data)
            if not isinstance(outputs, tuple):
                continue

            coarse_logits, fine_all = outputs

            # Coarse
            pred_c = torch.argmax(coarse_logits, 1)
            coarse_correct += (pred_c == tgts_1).sum().item()

            # Fine oracle
            fine_oracle = fine_all[torch.arange(B, device=data.device), tgts_1]
            pred_f_oracle = torch.argmax(fine_oracle, 1)
            fine_oracle_correct += (pred_f_oracle == tgts_2).sum().item()

            # Hard routing
            fine_hard = fine_all[torch.arange(B, device=data.device), pred_c]
            pred_f_hard = torch.argmax(fine_hard, 1)
            joint_hard = (pred_c == tgts_1) & (pred_f_hard == tgts_2)
            hard_correct += joint_hard.sum().item()

            # Soft routing (vectorized)
            coarse_prob = F.softmax(coarse_logits, dim=1)    # (B, G)
            fine_prob = F.softmax(fine_all, dim=2)            # (B, G, fpg)

            # P(class c) = coarse_prob[g(c)] * fine_prob[g(c), l(c)]
            class_prob = coarse_prob[:, coarse_idx] * fine_prob[:, coarse_idx, local_idx]
            pred_soft = torch.argmax(class_prob, dim=1)
            soft_correct += (pred_soft == targets).sum().item()

            total += B

    return {
        'hard_acc': hard_correct / total,
        'soft_acc': soft_correct / total,
        'coarse_acc': coarse_correct / total,
        'fine_oracle_acc': fine_oracle_correct / total,
        'total': total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    rank = args.rank
    torch.cuda.set_device(rank)
    seed_str = 'seed{:03d}'.format(args.seed)

    # Find all sequential model dirs
    model_dirs = sorted(glob.glob('output/cifar100/seq.*.R32'))
    if not model_dirs:
        print("No seq models found in output/cifar100/seq.*.R32")
        return

    print("=" * 70)
    print(" Soft Routing Evaluation")
    print("=" * 70)
    print("{:<35} {:>8} {:>8} {:>8} {:>8}".format(
        "Model", "Hard%", "Soft%", "Coarse%", "FineOr%"))
    print("-" * 70)

    cfg_path = 'configs/cifar100/seq_base.yaml'

    for model_dir in model_dirs:
        name = os.path.basename(model_dir)
        ckpt_path = os.path.join(model_dir, seed_str, 'models', 'best_model.pth')
        if not os.path.isfile(ckpt_path):
            continue

        # Determine hier_type from name
        if 'ProbW-NoSG' in name:
            hier_type = 'true_seq_probw_nosg'
        elif 'ProbW' in name:
            hier_type = 'true_seq_probw'
        elif 'Residual' in name:
            hier_type = 'true_seq_residual'
        else:
            hier_type = 'true_seq_direct'

        # Determine if random hierarchy
        random_hier = 'Rand' in name

        # Determine grouping file
        grouping = 'none'
        if 'KM' in name and 'Rand' not in name:
            grouping = 'groupings/seed{}/kmeans.json'.format(args.seed)
        elif 'Conf' in name:
            grouping = 'groupings/seed{}/confusion.json'.format(args.seed)

        # Build config
        class FakeArgs:
            def __init__(self):
                self.cfg = cfg_path
                self.opts = [
                    'dataset.hier_type', hier_type,
                    'dataset.random_hierarchy', str(random_hier),
                    'dataset.grouping_file', grouping,
                ]
        try:
            update_config(cfg, FakeArgs())
        except Exception:
            continue

        num_classes = cfg.dataset.num_classes
        model = get_model(cfg, num_classes, rank)

        try:
            ckpt = torch.load(ckpt_path, map_location='cuda:{}'.format(rank))
            state = {k.replace('module.', ''): v
                     for k, v in ckpt['state_dict'].items()}
            model.load_state_dict(state, strict=False)
        except Exception as e:
            print("{:<35} LOAD ERROR: {}".format(name, str(e)[:40]))
            continue

        mm = model.module if hasattr(model, 'module') else model
        if not hasattr(mm, 'is_true_sequential') or not mm.is_true_sequential:
            continue

        # Get f2c
        from builder.network import Network
        f2c = Network._get_fine_to_coarse(cfg)
        num_groups = len(set(f2c))

        # Dataset
        transform = get_transform(cfg, mode='test')
        if grouping != 'none':
            test_set = custom_dataset.UnsupHierCIFAR100(
                cfg, train=False, download=True, transform=transform)
        else:
            test_set = custom_dataset.HierCIFAR100(
                cfg, train=False, download=True, transform=transform)
        test_loader = DataLoader(
            test_set, batch_size=256, shuffle=False, num_workers=2)

        # Eval
        max_fpg = mm.max_fpg
        results = eval_model_fast(
            model, test_loader, f2c, num_groups, max_fpg, rank)

        delta = results['soft_acc'] - results['hard_acc']
        print("{:<35} {:>7.2f}% {:>7.2f}% {:>7.2f}% {:>7.2f}%  ({:+.2f}%)".format(
            name,
            results['hard_acc'] * 100,
            results['soft_acc'] * 100,
            results['coarse_acc'] * 100,
            results['fine_oracle_acc'] * 100,
            delta * 100))

    print("=" * 70)
    print("Flat baseline: 71.52%")
    print("Hard = argmax coarse → argmax fine (기존)")
    print("Soft = P(c) = P(g) × P(c|g) → argmax over 100 (제안)")
    print("=" * 70)


if __name__ == '__main__':
    main()
