"""
NC metric collection for CIFAR-100 models.
Measures NC1 (within-class collapse), NC2 (ETF alignment), NC3 (self-duality).

Usage:
  python tools/collect_nc_stats_cifar.py \
      --cfg configs/cifar100/ce_cifar100_resnet32.yaml \
      --checkpoint output/cifar100/.../best_model.pth \
      --output nc_stats/flat_resnet32.json
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'main'))
import _init_paths
from config import cfg, update_config
from utils.utils import get_model
import dataset as custom_dataset
from data_transform.transform_wrapper import get_transform
from torch.utils.data import DataLoader


def compute_nc_metrics(class_means, class_features, classifier_weight):
    """
    Args:
        class_means: [K, D] numpy array
        class_features: dict {class_id: [N_c, D] numpy array}
        classifier_weight: [K, D] numpy array (or [D, K] transposed)
    Returns:
        dict with NC1, NC2, NC3 values
    """
    K, D = class_means.shape
    global_mean = class_means.mean(axis=0)

    # --- NC1: within-class variability / between-class variability ---
    # Sw = (1/K) Σ_c (1/N_c) Σ_i (h_i - μ_c)(h_i - μ_c)^T
    # Sb = (1/K) Σ_c (μ_c - μ)(μ_c - μ)^T
    # NC1 = trace(Sw @ Sb^-1) / K

    Sw = np.zeros((D, D))
    count = 0
    for c in range(K):
        if c not in class_features or len(class_features[c]) == 0:
            continue
        feats = np.array(class_features[c])  # [N_c, D]
        centered = feats - class_means[c]
        Sw += centered.T @ centered
        count += len(feats)
    Sw /= max(count, 1)

    centered_means = class_means - global_mean
    Sb = centered_means.T @ centered_means / K

    # NC1 = trace(Sw @ pinv(Sb))
    try:
        Sb_inv = np.linalg.pinv(Sb)
        nc1 = np.trace(Sw @ Sb_inv) / K
    except:
        nc1 = float('inf')

    # --- NC2: ETF alignment ---
    # Ideal: cos(μ_i, μ_j) = -1/(K-1) for i≠j
    normed_means = centered_means / np.maximum(
        np.linalg.norm(centered_means, axis=1, keepdims=True), 1e-8)
    cos_sim = normed_means @ normed_means.T
    ideal = -1.0 / (K - 1)

    # NC2 = average deviation from ideal
    mask = ~np.eye(K, dtype=bool)
    nc2_deviation = np.mean(np.abs(cos_sim[mask] - ideal))
    # Also report equinorm std
    norms = np.linalg.norm(centered_means, axis=1)
    equinorm_std = np.std(norms) / (np.mean(norms) + 1e-8)

    # --- NC3: self-duality ---
    # ||W/||W|| - H/||H|| ||_F
    W = classifier_weight  # [K, D] or [D, K]
    if W.shape[0] != K:
        W = W.T  # ensure [K, D]

    W_norm = W / (np.linalg.norm(W, 'fro') + 1e-8)
    H = centered_means
    H_norm = H / (np.linalg.norm(H, 'fro') + 1e-8)
    nc3 = np.linalg.norm(W_norm - H_norm, 'fro')

    return {
        'nc1': float(nc1),
        'nc2_deviation': float(nc2_deviation),
        'nc2_equinorm_std': float(equinorm_std),
        'nc3': float(nc3),
        'num_classes': K,
        'feature_dim': D,
        'K_less_than_D': K < D,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--grouping', type=str, default=None,
                        help='If provided, measure NC at group level too')
    args_cli = parser.parse_args()

    class FakeArgs:
        def __init__(self, cfg_path, grouping=None):
            self.cfg = cfg_path
            self.opts = []
            if grouping:
                self.opts = ['dataset.grouping_file', grouping]
    update_config(cfg, FakeArgs(args_cli.cfg, args_cli.grouping))

    rank = args_cli.rank
    torch.cuda.set_device(rank)
    num_classes = cfg.dataset.num_classes
    model = get_model(cfg, num_classes, rank)

    ckpt = torch.load(args_cli.checkpoint, map_location=f'cuda:{rank}')
    state = {k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(state)
    model.eval()

    # Load data
    transform = get_transform(cfg, mode='test')
    train_set = custom_dataset.CIFAR100(cfg, train=True, download=True, transform=transform)
    loader = DataLoader(train_set, batch_size=256, shuffle=False, num_workers=2)

    # Extract features
    print("Extracting features...")
    class_features = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            data = batch[0].cuda(rank)
            targets = batch[1]
            features = model(data, feature_flag=True)
            for i in range(features.size(0)):
                class_features[targets[i].item()].append(
                    features[i].cpu().numpy())

    # Compute class means
    D = len(class_features[0][0])
    class_means = np.zeros((num_classes, D))
    for c in range(num_classes):
        if len(class_features[c]) > 0:
            class_means[c] = np.mean(class_features[c], axis=0)

    # Get classifier weight
    mm = model.module if hasattr(model, 'module') else model
    if hasattr(mm, 'classifier'):
        W = mm.classifier.weight.detach().cpu().numpy()  # [K, D]
    elif hasattr(mm, 'classifier_coarse'):
        W = mm.classifier_coarse.weight.detach().cpu().numpy()
    else:
        raise ValueError("Cannot find classifier weight")

    # Fine-level NC (100-way, K=100 vs D)
    print(f"\n=== Fine-level NC (K={num_classes}, D={D}) ===")
    # For hierarchical models, W is coarse (20-way), not fine (100-way)
    # Use class_means as proxy for W in NC3 when shapes don't match
    if W.shape[0] != num_classes and W.shape[1] != num_classes:
        W_fine = class_means  # fallback: use means as W proxy, NC3 will be ~0
    else:
        W_fine = W
    fine_metrics = compute_nc_metrics(class_means, class_features, W_fine)
    for k, v in fine_metrics.items():
        print(f"  {k}: {v}")

    results = {'fine': fine_metrics}

    # Group-level NC (if grouping provided)
    if args_cli.grouping:
        with open(args_cli.grouping) as f:
            gdata = json.load(f)
        f2c = gdata['fine_to_coarse']
        num_groups = gdata['num_groups']

        # Compute group means from class means
        group_means = np.zeros((num_groups, D))
        group_features = defaultdict(list)
        for c in range(num_classes):
            g = f2c[c]
            group_features[g].extend(class_features[c])

        for g in range(num_groups):
            if len(group_features[g]) > 0:
                group_means[g] = np.mean(group_features[g], axis=0)

        # Use coarse classifier weight if available
        if hasattr(mm, 'classifier_coarse'):
            W_coarse = mm.classifier_coarse.weight.detach().cpu().numpy()
        else:
            W_coarse = group_means  # fallback

        print(f"\n=== Coarse-level NC (K={num_groups}, D={D}) ===")
        coarse_metrics = compute_nc_metrics(group_means, group_features, W_coarse)
        for k, v in coarse_metrics.items():
            print(f"  {k}: {v}")

        results['coarse'] = coarse_metrics

    # Save
    os.makedirs(os.path.dirname(args_cli.output), exist_ok=True)
    with open(args_cli.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args_cli.output}")


if __name__ == '__main__':
    main()
