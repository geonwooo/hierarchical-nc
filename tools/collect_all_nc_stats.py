#!/usr/bin/env python
"""
collect_all_nc_stats.py — seq.* + v4.* 모델 전부의 NC stats 수집
Usage: python tools/collect_all_nc_stats.py --rank 0
"""
import argparse, glob, json, os, sys
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


def name_to_opts(name):
    """Model name → config opts 매핑"""
    opts = ['dataset.hier_type', 'true_seq_direct']

    # hier_type
    if 'Residual' in name or 'Res-' in name or 'Res.' in name:
        opts[1] = 'true_seq_residual'
    elif 'ProbW-NoSG' in name:
        opts[1] = 'true_seq_probw_nosg'
    elif 'ProbW' in name:
        opts[1] = 'true_seq_probw'

    # Grouping
    if 'Rand' in name:
        opts += ['dataset.random_hierarchy', 'True']
    elif 'KM' in name and 'Rand' not in name:
        opts += ['dataset.grouping_file', 'groupings/seed0/kmeans.json']
    elif 'Conf' in name and 'v4' not in name:
        opts += ['dataset.grouping_file', 'groupings/seed0/confusion.json']

    # V4 options
    if 'Cos' in name:
        opts += ['loss.cosine_classifier', 'True']
        if 'Cos32' in name:
            opts += ['loss.cosine_scale', '32.0']
        else:
            opts += ['loss.cosine_scale', '16.0']

    if 'MLP-H16' in name:
        opts += ['loss.fine_hidden', '16']
    elif 'MLP-H64' in name:
        opts += ['loss.fine_hidden', '64']
    elif 'MLP-H128' in name:
        opts += ['loss.fine_hidden', '128']
    elif 'MLP32' in name or 'MLP-H32' in name:
        opts += ['loss.fine_hidden', '32']
    elif 'MLP64' in name:
        opts += ['loss.fine_hidden', '64']

    if 'FiLM' in name:
        opts += ['loss.use_film', 'True']
    if 'ETFc' in name:
        opts += ['loss.etf_coarse', 'True']
    if 'ETFcf' in name or 'ETFf' in name:
        opts += ['loss.etf_fine', 'True']
    if 'J100' in name:
        opts += ['loss.joint_100way', 'True']

    # DualObj
    if 'DualObj-S1' in name:
        opts += ['loss.two_stage_mode', 'coarse_only']

    # Flat
    if name.startswith('flat') or name.startswith('Flat') or name.startswith('CE.CIFAR'):
        opts = ['dataset.hier_type', 'flat']

    return opts


def compute_nc_stats(features_by_class, classifier_weight=None):
    """Compute NC1, NC2, NC3 from features grouped by class"""
    classes = sorted(features_by_class.keys())
    K = len(classes)
    if K < 2:
        return {'nc1': 0, 'nc2_deviation': 0, 'nc2_equinorm_std': 0, 'nc3': 0}

    # Class means
    means = []
    for c in classes:
        feats = features_by_class[c]
        means.append(feats.mean(dim=0))
    means = torch.stack(means)  # (K, D)
    global_mean = means.mean(dim=0)  # (D,)
    D = means.shape[1]

    # NC1: within-class variability / between-class variability
    # Sw = avg within-class covariance, Sb = between-class covariance
    Sw = torch.zeros(D, D)
    for i, c in enumerate(classes):
        feats = features_by_class[c]
        centered = feats - means[i]
        Sw += (centered.T @ centered) / feats.shape[0]
    Sw /= K

    centered_means = means - global_mean
    Sb = (centered_means.T @ centered_means) / K

    # NC1 = Tr(Sw @ Sb^{-1}) / K
    try:
        Sb_inv = torch.linalg.pinv(Sb)
        nc1 = torch.trace(Sw @ Sb_inv).item() / K
    except:
        nc1 = float('inf')

    # NC2: equiangularity
    means_centered = means - global_mean
    means_norm = F.normalize(means_centered, dim=1)
    cos_sim = means_norm @ means_norm.T  # (K, K)
    mask = ~torch.eye(K, dtype=torch.bool)
    off_diag = cos_sim[mask]
    target = -1.0 / (K - 1)
    nc2_deviation = (off_diag - target).abs().mean().item()

    # equinorm std
    norms = means_centered.norm(dim=1)
    nc2_equinorm_std = (norms.std() / norms.mean()).item() if norms.mean() > 0 else 0

    # NC3: self-duality (W ≈ class means)
    nc3 = 0.0
    if classifier_weight is not None and classifier_weight.shape[0] == K:
        W_norm = F.normalize(classifier_weight, dim=1)
        H_norm = F.normalize(means_centered, dim=1)
        cos_WH = (W_norm * H_norm).sum(dim=1)  # per-class cosine
        nc3 = cos_WH.mean().item()

    return {
        'nc1': nc1,
        'nc2_deviation': nc2_deviation,
        'nc2_equinorm_std': nc2_equinorm_std,
        'nc3': nc3,
        'num_classes': K,
        'feature_dim': D,
        'K_less_than_D': K < D,
    }


def compute_ncm_acc(train_features_by_class, test_features, test_labels):
    """NCM accuracy: classify by nearest class mean"""
    classes = sorted(train_features_by_class.keys())
    means = torch.stack([train_features_by_class[c].mean(dim=0) for c in classes])
    means_norm = F.normalize(means, dim=1)

    test_norm = F.normalize(test_features, dim=1)
    sim = test_norm @ means_norm.T
    pred = torch.tensor(classes)[sim.argmax(dim=1)]
    acc = (pred == test_labels).float().mean().item()
    return acc


def process_model(model_dir, rank, cfg_path='configs/cifar100/seq_base.yaml'):
    name = os.path.basename(model_dir)

    # Find checkpoint
    ckpt_path = None
    for p in [
        os.path.join(model_dir, 'seed000/models/best_model.pth'),
        os.path.join(model_dir, 'models/best_model.pth'),
    ]:
        if os.path.isfile(p):
            ckpt_path = p
            break
    if not ckpt_path:
        return None

    # Config
    opts = name_to_opts(name)

    # Handle flat model differently
    is_flat = 'flat' in opts

    class FakeArgs:
        def __init__(self, cfg_p, o):
            self.cfg = cfg_p
            self.opts = o

    try:
        update_config(cfg, FakeArgs(cfg_path, opts))
    except:
        return None

    torch.cuda.set_device(rank)
    num_classes = cfg.dataset.num_classes

    try:
        model = get_model(cfg, num_classes, rank)
        ckpt = torch.load(ckpt_path, map_location=f'cuda:{rank}', weights_only=False)
        state = {k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()}
        model.load_state_dict(state, strict=False)
        model.eval()
    except Exception as e:
        return {'name': name, 'error': str(e)[:60]}

    mm = model.module if hasattr(model, 'module') else model

    # Dataset - use train for class means, test for eval
    transform_train = get_transform(cfg, mode='test')  # no aug for feature extraction
    transform_test = get_transform(cfg, mode='test')

    try:
        grouping = getattr(cfg.dataset, 'grouping_file', 'none')
        if grouping and grouping != 'none':
            train_set = custom_dataset.UnsupHierCIFAR100(cfg, train=True, download=True, transform=transform_train)
            test_set = custom_dataset.UnsupHierCIFAR100(cfg, train=False, download=True, transform=transform_test)
        elif is_flat:
            from torchvision import datasets, transforms
            norm = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            tf = transforms.Compose([transforms.ToTensor(), norm])
            train_set = datasets.CIFAR100('/data/hoyong', train=True, download=True, transform=tf)
            test_set = datasets.CIFAR100('/data/hoyong', train=False, download=True, transform=tf)
        else:
            train_set = custom_dataset.HierCIFAR100(cfg, train=True, download=True, transform=transform_train)
            test_set = custom_dataset.HierCIFAR100(cfg, train=False, download=True, transform=transform_test)
    except:
        return {'name': name, 'error': 'dataset load failed'}

    train_loader = DataLoader(train_set, batch_size=256, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

    # Extract features
    train_features_100 = defaultdict(list)  # by fine class
    train_features_20 = defaultdict(list)   # by coarse group
    test_features_list = []
    test_labels_list = []

    f2c = FINE_TO_COARSE

    with torch.no_grad():
        for batch in train_loader:
            data = batch[0].cuda(rank)
            if is_flat:
                targets = batch[1]
            else:
                targets = batch[1]  # fine class
            h = mm.extract_feature(data).cpu()
            for i in range(h.size(0)):
                c = targets[i].item()
                train_features_100[c].append(h[i])
                train_features_20[f2c[c]].append(h[i])

        for batch in test_loader:
            data = batch[0].cuda(rank)
            if is_flat:
                targets = batch[1]
            else:
                targets = batch[1]
            h = mm.extract_feature(data).cpu()
            test_features_list.append(h)
            test_labels_list.append(targets)

    # Stack
    for c in train_features_100:
        train_features_100[c] = torch.stack(train_features_100[c])
    for g in train_features_20:
        train_features_20[g] = torch.stack(train_features_20[g])
    test_features = torch.cat(test_features_list)
    test_labels = torch.cat(test_labels_list)

    result = {'name': name}

    # Fine-level NC (100 classes)
    fine_w = None
    if hasattr(mm, 'classifier') and hasattr(mm.classifier, 'weight'):
        fine_w = mm.classifier.weight.data.cpu()
    result['fine'] = compute_nc_stats(train_features_100, fine_w)

    # NCM accuracy (100-way)
    result['fine_ncm_acc'] = compute_ncm_acc(train_features_100, test_features, test_labels)

    # Coarse-level NC (20 groups)
    coarse_w = None
    if hasattr(mm, 'classifier_coarse') and hasattr(mm.classifier_coarse, 'weight'):
        coarse_w = mm.classifier_coarse.weight.data.cpu()
    elif hasattr(mm, 'coarse_weight'):
        coarse_w = mm.coarse_weight.data.cpu()
    elif hasattr(mm, 'coarse_etf'):
        coarse_w = mm.coarse_etf.data.cpu()

    if coarse_w is not None or not is_flat:
        result['coarse'] = compute_nc_stats(train_features_20, coarse_w)
        # Coarse NCM
        coarse_labels = torch.tensor([f2c[t.item()] for t in test_labels])
        result['coarse_ncm_acc'] = compute_ncm_acc(train_features_20, test_features, coarse_labels)

    # Per-group fine NC (K=5 each, this is the KEY metric for HNC)
    if not is_flat:
        groups = defaultdict(lambda: defaultdict(list))
        for c in train_features_100:
            g = f2c[c]
            groups[g][c] = train_features_100[c]

        per_group_nc1 = []
        per_group_nc3 = []
        for g in sorted(groups.keys()):
            group_features = groups[g]
            if len(group_features) < 2:
                continue
            # Get per-group fine weight if available
            fine_w_g = None
            if hasattr(mm, 'fine_weight'):
                fine_w_g = mm.fine_weight.data[:, g, :].T.cpu()  # (max_fpg, D)
            elif hasattr(mm, 'fine_w1'):
                pass  # MLP, skip NC3 for fine

            stats = compute_nc_stats(group_features, fine_w_g)
            per_group_nc1.append(stats['nc1'])
            if stats['nc3'] != 0:
                per_group_nc3.append(stats['nc3'])

        if per_group_nc1:
            result['per_group_fine'] = {
                'mean_nc1': float(np.mean(per_group_nc1)),
                'std_nc1': float(np.std(per_group_nc1)),
                'mean_nc3': float(np.mean(per_group_nc3)) if per_group_nc3 else 0,
                'num_groups': len(per_group_nc1),
                'K_per_group': 5,
                'feature_dim': 64,
                'K_less_than_D': True,
            }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, default=0)
    args = parser.parse_args()

    # Find all model dirs
    model_dirs = []

    # Flat
    flat_dir = 'output/cifar100/CE.CIFAR100.ResNet32.200epoch'
    if os.path.isdir(flat_dir):
        model_dirs.append(flat_dir)

    # Seq models
    for d in sorted(glob.glob('output/cifar100/seq.*.R32')):
        model_dirs.append(d)

    # V3 models
    for d in sorted(glob.glob("output/cifar100/v3.*.R32")):
        model_dirs.append(d)

    # V4 models
    for d in sorted(glob.glob('output/cifar100/v4.*.R32')):
        model_dirs.append(d)

    print("=" * 90)
    print(f" NC Stats Collection — {len(model_dirs)} models")
    print("=" * 90)
    print(f"{'Model':<35} {'Fine NC1':>8} {'Fine NC3':>8} {'Coarse NC1':>10} {'Coarse NC3':>10} {'PG-Fine NC1':>11} {'NCM 100':>8}")
    print("-" * 90)

    all_results = {}
    for model_dir in model_dirs:
        name = os.path.basename(model_dir)
        result = process_model(model_dir, args.rank)
        if result is None:
            print(f"{name:<35} NO CHECKPOINT")
            continue
        if 'error' in result:
            print(f"{name:<35} ERROR: {result['error']}")
            continue

        fine_nc1 = f"{result['fine']['nc1']:.3f}"
        fine_nc3 = f"{result['fine']['nc3']:.3f}"
        coarse_nc1 = f"{result.get('coarse', {}).get('nc1', '')}"
        if coarse_nc1:
            coarse_nc1 = f"{float(coarse_nc1):.3f}"
        coarse_nc3 = f"{result.get('coarse', {}).get('nc3', '')}"
        if coarse_nc3:
            coarse_nc3 = f"{float(coarse_nc3):.3f}"
        pg_nc1 = f"{result.get('per_group_fine', {}).get('mean_nc1', '')}"
        if pg_nc1:
            pg_nc1 = f"{float(pg_nc1):.3f}"
        ncm = f"{result.get('fine_ncm_acc', 0)*100:.2f}%"

        print(f"{name:<35} {fine_nc1:>8} {fine_nc3:>8} {coarse_nc1:>10} {coarse_nc3:>10} {pg_nc1:>11} {ncm:>8}")

        all_results[name] = result

        # Save individual
        out_path = f"nc_stats/seed0/seq/{name}.json"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # Convert numpy to python types
        def convert(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
        with open(out_path, 'w') as f:
            json.dump(convert(result), f, indent=2)

    # Save all
    print("=" * 90)
    print(f"Saved {len(all_results)} NC stats to nc_stats/seed0/seq/")


if __name__ == '__main__':
    main()
