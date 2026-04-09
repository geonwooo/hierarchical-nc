"""
Unsupervised hierarchy discovery for CIFAR-100.
Discovers 20-group hierarchy from learned features without using CIFAR-100 coarse labels.

Usage:
  # 1. Train flat baseline first
  bash TrainResNet32CIFAR100.sh 0 flat_base 0

  # 2. Discover hierarchy
  python tools/discover_hierarchy.py \
      --method kmeans \
      --checkpoint output/cifar100/CE.CIFAR100.ResNet32.200epoch/seed000/models/best_model.pth \
      --cfg configs/cifar100/ce_cifar100_resnet32.yaml \
      --num-groups 20 \
      --output groupings/unsup_kmeans_20.json

  # 3. Re-train with discovered hierarchy
  python main/train.py --cfg configs/cifar100/v3_unsup_kmeans.yaml ...
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'main'))
import _init_paths
from config import cfg, update_config
from utils.utils import get_model
import dataset as custom_dataset
from data_transform.transform_wrapper import get_transform
from torch.utils.data import DataLoader


def extract_class_means(model, dataloader, num_classes, rank=0):
    """Extract per-class feature means from trained model."""
    model.eval()
    class_features = defaultdict(list)

    with torch.no_grad():
        for batch in dataloader:
            data, targets = batch[0].cuda(rank), batch[1]
            features = model(data, feature_flag=True)  # [B, D]
            for i in range(features.size(0)):
                class_features[targets[i].item()].append(
                    features[i].cpu().numpy())

    # Compute means
    D = features.size(1)
    class_means = np.zeros((num_classes, D))
    for c in range(num_classes):
        if len(class_features[c]) > 0:
            class_means[c] = np.mean(class_features[c], axis=0)

    return class_means


def extract_confusion_matrix(model, dataloader, num_classes, rank=0):
    """Extract confusion matrix from trained model."""
    model.eval()
    confusion = np.zeros((num_classes, num_classes))

    with torch.no_grad():
        for batch in dataloader:
            data, targets = batch[0].cuda(rank), batch[1].numpy()
            outputs = model(data)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            preds = torch.argmax(outputs, 1).cpu().numpy()
            for t, p in zip(targets, preds):
                confusion[t, p] += 1

    return confusion


# ============================================================
# Method 1: Balanced K-means on class means
# ============================================================
def discover_kmeans(class_means, num_groups, seed=42):
    """Balanced k-means: each group gets exactly N/K classes."""
    num_classes = class_means.shape[0]
    target_size = num_classes // num_groups  # 100/20 = 5

    # L2 normalize
    norms = np.linalg.norm(class_means, axis=1, keepdims=True)
    normed = class_means / np.maximum(norms, 1e-8)

    # Step 1: standard k-means for centroids
    km = KMeans(n_clusters=num_groups, random_state=seed, n_init=20)
    km.fit(normed)
    centroids = km.cluster_centers_
    c_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.maximum(c_norms, 1e-8)

    # Step 2: balanced assignment via cost matrix + linear_sum_assignment
    # Cost = 1 - cosine_sim, replicated for balanced constraint
    sim = normed @ centroids.T  # [100, 20]
    cost = 1.0 - sim  # [100, 20]

    # Replicate columns: each group appears target_size times
    # [100, 20*5=100] → square matrix for Hungarian
    cost_rep = np.repeat(cost, target_size, axis=1)  # [100, 100]
    row_ind, col_ind = linear_sum_assignment(cost_rep)
    assignments = col_ind // target_size  # map back to group index

    # Verify balance
    unique, counts = np.unique(assignments, return_counts=True)
    print(f"  Balanced k-means: {len(unique)} groups, sizes: min={counts.min()}, max={counts.max()}")

    return assignments


# ============================================================
# Method 2: Confusion-guided clustering
# ============================================================
def discover_confusion(confusion_matrix, num_groups):
    """Cluster classes that are frequently confused with each other."""
    num_classes = confusion_matrix.shape[0]

    # Symmetrize: similarity = confusion[i,j] + confusion[j,i]
    sim = confusion_matrix + confusion_matrix.T
    # Normalize rows
    row_sums = sim.sum(axis=1, keepdims=True)
    sim_norm = sim / np.maximum(row_sums, 1e-8)

    # Convert similarity to distance
    dist = 1.0 - sim_norm

    # Agglomerative clustering using sklearn
    from sklearn.cluster import AgglomerativeClustering
    agg = AgglomerativeClustering(
        n_clusters=num_groups,
        metric='precomputed',
        linkage='average'
    )
    assignments = agg.fit_predict(dist)

    return assignments


# ============================================================
# Method 3: Optimal Transport (Sinkhorn) balanced assignment
# ============================================================
def discover_ot(class_means, num_groups, num_iters=100, epsilon=0.05):
    """Balanced assignment via Sinkhorn-Knopp optimal transport."""
    num_classes = class_means.shape[0]
    target_size = num_classes // num_groups  # 100/20 = 5

    # L2 normalize
    norms = np.linalg.norm(class_means, axis=1, keepdims=True)
    normed = class_means / np.maximum(norms, 1e-8)

    # Initialize prototypes via k-means
    km = KMeans(n_clusters=num_groups, random_state=42, n_init=10)
    km.fit(normed)
    prototypes = km.cluster_centers_
    proto_norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    prototypes = prototypes / np.maximum(proto_norms, 1e-8)

    # Similarity matrix: [num_classes, num_groups]
    S = normed @ prototypes.T  # cosine similarity

    # Sinkhorn-Knopp for balanced assignment
    Q = np.exp(S / epsilon)  # [100, 20]

    for _ in range(num_iters):
        # Row normalization (each class sums to 1)
        Q = Q / Q.sum(axis=1, keepdims=True)
        # Column normalization (each group gets target_size classes)
        col_sums = Q.sum(axis=0, keepdims=True)
        Q = Q * (target_size / np.maximum(col_sums, 1e-8))

    # Hard assignment from soft Q
    assignments = np.argmax(Q, axis=1)

    return assignments


# ============================================================
# Method 4: Random (control)
# ============================================================
def discover_random(num_classes, num_groups, seed=42):
    """Random balanced assignment (control)."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(num_classes)
    assignments = np.zeros(num_classes, dtype=np.int64)
    per_group = num_classes // num_groups

    for g in range(num_groups):
        start = g * per_group
        end = start + per_group if g < num_groups - 1 else num_classes
        assignments[perm[start:end]] = g

    return assignments


# ============================================================
# Utility: save/load grouping
# ============================================================
def save_grouping(filepath, method, assignments, class_means=None):
    """Save discovered grouping to JSON."""
    num_classes = len(assignments)
    fine_to_coarse = [int(g) for g in assignments]

    # Build group_to_fine
    group_to_fine = defaultdict(list)
    for c, g in enumerate(fine_to_coarse):
        group_to_fine[g].append(c)

    # Stats
    sizes = [len(group_to_fine[g]) for g in sorted(group_to_fine.keys())]

    data = {
        "method": method,
        "num_classes": num_classes,
        "num_groups": len(set(fine_to_coarse)),
        "fine_to_coarse": fine_to_coarse,
        "group_to_fine": {str(k): v for k, v in group_to_fine.items()},
        "group_sizes": sizes,
    }

    # Save centroids if available
    if class_means is not None:
        num_groups = len(set(fine_to_coarse))
        centroids = np.zeros((num_groups, class_means.shape[1]))
        for g in range(num_groups):
            members = [c for c, gg in enumerate(fine_to_coarse) if gg == g]
            if len(members) > 0:
                centroids[g] = class_means[members].mean(axis=0)
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        centroids = centroids / np.maximum(norms, 1e-8)
        data["centroids"] = centroids.tolist()

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"  Saved: {filepath}")
    print(f"  Groups: {len(set(fine_to_coarse))}, sizes: {sizes}")

    return fine_to_coarse


def evaluate_grouping(fine_to_coarse, gt_fine_to_coarse=None):
    """Evaluate grouping quality against ground truth (if available)."""
    if gt_fine_to_coarse is None:
        return

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    ari = adjusted_rand_score(gt_fine_to_coarse, fine_to_coarse)
    nmi = normalized_mutual_info_score(gt_fine_to_coarse, fine_to_coarse)
    print(f"  vs GT: ARI={ari:.4f}, NMI={nmi:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True,
                        choices=['kmeans', 'confusion', 'ot', 'random'])
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained flat model checkpoint')
    parser.add_argument('--cfg', type=str, required=True,
                        help='Config file for the flat model')
    parser.add_argument('--num-groups', type=int, default=20)
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON path')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rank', type=int, default=0)
    args_cli = parser.parse_args()

    # Build config
    class FakeArgs:
        def __init__(self, cfg):
            self.cfg = cfg
            self.opts = []
    fake_args = FakeArgs(args_cli.cfg)
    update_config(cfg, fake_args)

    # Load model
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

    print(f"Method: {args_cli.method}, Groups: {args_cli.num_groups}")

    # Extract features
    print("Extracting class means...")
    class_means = extract_class_means(model, loader, num_classes, rank)
    print(f"  Class means shape: {class_means.shape}")

    # Discover grouping
    if args_cli.method == 'kmeans':
        assignments = discover_kmeans(class_means, args_cli.num_groups, args_cli.seed)
    elif args_cli.method == 'confusion':
        print("Computing confusion matrix...")
        confusion = extract_confusion_matrix(model, loader, num_classes, rank)
        assignments = discover_confusion(confusion, args_cli.num_groups)
    elif args_cli.method == 'ot':
        assignments = discover_ot(class_means, args_cli.num_groups)
    elif args_cli.method == 'random':
        assignments = discover_random(num_classes, args_cli.num_groups, args_cli.seed)

    # Save
    fine_to_coarse = save_grouping(
        args_cli.output, args_cli.method, assignments, class_means)

    # Evaluate against GT
    from dataset.hier import HierCIFAR100
    gt = HierCIFAR100.FINE_TO_COARSE
    evaluate_grouping(fine_to_coarse, gt)

    print("Done!")


if __name__ == '__main__':
    main()
