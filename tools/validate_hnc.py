#!/usr/bin/env python
"""
HNC Validation: Layer-wise NC Analysis + Intermediate Concat
=============================================================
목적:
  1. Prerequisite 1: D^L < K-1일 때 intermediate layer에서 NC 발생하는가?
  2. Prerequisite 2: Concat으로 D를 늘리면 NC가 발생하는가?
  3. 3 Cases NC metric 측정
  4. Semantic hierarchy 검증 (D^L에서 grouping이 GT superclass와 일치?)

Usage:
  python tools/validate_hnc.py --rank 0

입력: 기존 Flat checkpoint (학습 불필요)
출력: JSON + 터미널 리포트
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'main'))


# ============================================================
# NC Metrics
# ============================================================
def compute_nc_metrics(features_by_class, classifier_weight=None):
    """Compute NC1, NC2, NC3, NCC accuracy from features grouped by class"""
    classes = sorted(features_by_class.keys())
    K = len(classes)
    if K < 2:
        return {}

    means = []
    for c in classes:
        means.append(features_by_class[c].mean(dim=0))
    means = torch.stack(means)
    global_mean = means.mean(dim=0)
    D = means.shape[1]

    # NC1
    Sw = torch.zeros(D, D)
    for i, c in enumerate(classes):
        feats = features_by_class[c]
        centered = feats - means[i]
        Sw += (centered.T @ centered) / feats.shape[0]
    Sw /= K
    centered_means = means - global_mean
    Sb = (centered_means.T @ centered_means) / K
    try:
        Sb_inv = torch.linalg.pinv(Sb)
        nc1 = torch.trace(Sw @ Sb_inv).item() / K
    except:
        nc1 = float('inf')

    # NC2
    means_norm = F.normalize(centered_means, dim=1)
    cos_sim = means_norm @ means_norm.T
    mask = ~torch.eye(K, dtype=torch.bool)
    off_diag = cos_sim[mask]
    target = -1.0 / (K - 1)
    nc2 = (off_diag - target).abs().mean().item()

    norms = centered_means.norm(dim=1)
    equinorm_std = (norms.std() / norms.mean()).item() if norms.mean() > 0 else 0

    # NC3
    nc3 = 0.0
    if classifier_weight is not None and classifier_weight.shape[0] == K:
        W_norm = F.normalize(classifier_weight, dim=1)
        H_norm = F.normalize(centered_means, dim=1)
        nc3 = (W_norm * H_norm).sum(dim=1).mean().item()

    return {
        'nc1': round(nc1, 4),
        'nc2': round(nc2, 4),
        'nc2_equinorm_std': round(equinorm_std, 4),
        'nc3': round(nc3, 4),
        'num_classes': K,
        'feature_dim': D,
        'K_less_than_D': K - 1 < D,
    }


def compute_ncc_accuracy(features_by_class, test_features, test_labels):
    """NCC accuracy"""
    classes = sorted(features_by_class.keys())
    means = torch.stack([features_by_class[c].mean(dim=0) for c in classes])
    means_norm = F.normalize(means, dim=1)
    test_norm = F.normalize(test_features, dim=1)
    sim = test_norm @ means_norm.T
    pred = torch.tensor(classes)[sim.argmax(dim=1)]
    return (pred == test_labels).float().mean().item()


# ============================================================
# Feature Extraction (ResNet32 layer-wise)
# ============================================================
class LayerFeatureExtractor:
    """ResNet32의 각 layer에서 feature를 추출"""
    def __init__(self, model):
        self.model = model
        self.features = {}
        self._register_hooks()

    def _register_hooks(self):
        mm = self.model.module if hasattr(self.model, 'module') else self.model
        backbone = mm.backbone

        # ResNet32 구조: conv1 → layer1 → layer2 → layer3
        # layer1 output: 16 channels → GAP → 16-dim (too small)
        # layer2 output: 32 channels → GAP → 32-dim
        # layer3 output: 64 channels → GAP → 64-dim (last)
        # 하지만 실제로는 in_channels=16이면:
        #   layer1: 16ch, layer2: 32ch, layer3: 64ch

        def make_hook(name):
            def hook(module, input, output):
                # GAP를 직접 적용
                if len(output.shape) == 4:  # conv output (B, C, H, W)
                    pooled = F.adaptive_avg_pool2d(output, 1).squeeze(-1).squeeze(-1)
                else:
                    pooled = output
                self.features[name] = pooled.detach().cpu()
            return hook

        backbone.layer1.register_forward_hook(make_hook('layer1'))
        backbone.layer2.register_forward_hook(make_hook('layer2'))
        backbone.layer3.register_forward_hook(make_hook('layer3'))

    def extract(self, dataloader, device):
        self.model.eval()
        mm = self.model.module if hasattr(self.model, 'module') else self.model

        all_features = {name: [] for name in ['layer1', 'layer2', 'layer3', 'last']}
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                data = batch[0].to(device)
                labels = batch[1]

                # Forward to trigger hooks
                h = mm.extract_feature(data)
                all_features['last'].append(h.cpu())

                for name in ['layer1', 'layer2', 'layer3']:
                    if name in self.features:
                        all_features[name].append(self.features[name])

                all_labels.append(labels)

        result = {}
        for name in all_features:
            if all_features[name]:
                result[name] = torch.cat(all_features[name], dim=0)
        labels = torch.cat(all_labels, dim=0)

        return result, labels


# ============================================================
# Semantic Hierarchy Analysis
# ============================================================
def analyze_hierarchy(class_means, gt_coarse_labels, layer_name):
    """Class mean들을 clustering하고 GT hierarchy와 비교"""
    K = class_means.shape[0]

    # Hierarchical clustering on class means
    means_np = class_means.numpy()
    # Cosine distance
    means_norm = means_np / (np.linalg.norm(means_np, axis=1, keepdims=True) + 1e-8)
    cos_dist = 1 - means_norm @ means_norm.T
    np.fill_diagonal(cos_dist, 0)

    # Condensed distance matrix
    from scipy.spatial.distance import squareform
    condensed = squareform(cos_dist)
    Z = linkage(condensed, method='ward')

    # Cut into 20 groups (CIFAR-100 has 20 superclasses)
    pred_groups = fcluster(Z, t=20, criterion='maxclust')

    # Compare with GT
    nmi = normalized_mutual_info_score(gt_coarse_labels, pred_groups)
    ari = adjusted_rand_score(gt_coarse_labels, pred_groups)

    return {
        'layer': layer_name,
        'NMI_vs_GT': round(nmi, 4),
        'ARI_vs_GT': round(ari, 4),
    }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--checkpoint', type=str,
                        default='output/cifar100/CE.CIFAR100.ResNet32.200epoch/seed000/models/best_model.pth')
    args = parser.parse_args()

    device = f'cuda:{args.rank}'
    torch.cuda.set_device(args.rank)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------
    print("=" * 80)
    print(" HNC Validation: Layer-wise NC + Concat Analysis")
    print("=" * 80)

    import _init_paths
    from config import cfg, update_config
    from utils.utils import get_model

    class FakeArgs:
        def __init__(self):
            self.cfg = 'configs/cifar100/seq_base.yaml'
            self.opts = ['dataset.hier_type', 'flat']

    update_config(cfg, FakeArgs())
    model = get_model(cfg, 100, args.rank)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = {k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    mm = model.module if hasattr(model, 'module') else model

    # Get classifier weight
    classifier_weight = mm.classifier.weight.data.cpu()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    norm = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    tf = transforms.Compose([transforms.ToTensor(), norm])
    train_set = datasets.CIFAR100('/data/hoyong', train=True, download=True, transform=tf)
    test_set = datasets.CIFAR100('/data/hoyong', train=False, download=True, transform=tf)
    train_loader = DataLoader(train_set, batch_size=256, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

    # CIFAR-100 GT coarse labels
    FINE_TO_COARSE = [
        4,1,14,8,0,6,7,7,18,3,3,14,9,18,7,11,3,9,7,11,6,11,5,10,7,
        6,13,15,3,15,0,11,1,10,12,14,16,9,11,5,5,19,8,8,15,13,14,17,
        18,10,16,4,17,4,2,0,17,4,18,17,10,3,2,12,12,16,12,1,9,19,2,
        10,0,1,16,12,9,13,15,13,16,19,2,4,6,19,5,5,8,19,18,1,2,15,
        6,0,17,8,14,13
    ]

    # --------------------------------------------------------
    # Extract features from all layers
    # --------------------------------------------------------
    print("\n>>> Extracting features from all layers...")
    extractor = LayerFeatureExtractor(model)

    # Train set
    train_features, train_labels = extractor.extract(train_loader, device)

    # Test set
    test_features_dict, test_labels = extractor.extract(test_loader, device)

    print(f"  Layers found: {list(train_features.keys())}")
    for name, feat in train_features.items():
        print(f"  {name}: shape={feat.shape}")

    # --------------------------------------------------------
    # Prerequisite 1: Layer-wise NC (D^L < K-1에서 intermediate NC 발생하는가?)
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print(" Prerequisite 1: Layer-wise NC Metrics")
    print(f" K=100 classes, Layers: {list(train_features.keys())}")
    print("=" * 80)

    results = {}

    for layer_name, features in train_features.items():
        D = features.shape[1]
        # Group by class
        feat_by_class = defaultdict(list)
        for i in range(features.shape[0]):
            c = train_labels[i].item()
            feat_by_class[c].append(features[i])
        for c in feat_by_class:
            feat_by_class[c] = torch.stack(feat_by_class[c])

        # NC metrics
        w = classifier_weight if layer_name == 'last' else None
        nc = compute_nc_metrics(feat_by_class, w)

        # NCC accuracy
        test_feat = test_features_dict[layer_name]
        ncc_acc = compute_ncc_accuracy(feat_by_class, test_feat, test_labels)
        nc['ncc_accuracy'] = round(ncc_acc * 100, 2)

        # Linear accuracy (only for last layer, known)
        if layer_name == 'last':
            nc['linear_accuracy'] = 71.52
            nc['linear_ncc_gap'] = round(71.52 - ncc_acc * 100, 2)

        results[layer_name] = nc

        k_vs_d = "K<D ✅ NC 가능" if D >= 100 else "K>D ❌ NC 제한"
        print(f"\n  [{layer_name}] D={D}, K=100 → {k_vs_d}")
        print(f"    NC1={nc['nc1']:.4f}  NC2={nc['nc2']:.4f}  NC3={nc['nc3']:.4f}")
        print(f"    NCC Acc={nc['ncc_accuracy']:.2f}%")
        if 'linear_ncc_gap' in nc:
            print(f"    Linear-NCC Gap={nc['linear_ncc_gap']:.2f}%")

    # --------------------------------------------------------
    # Prerequisite 2: Concat features → NC 발생하는가?
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print(" Prerequisite 2: Concat Features NC")
    print("=" * 80)

    concat_configs = []

    # 가능한 모든 concat 조합
    layer_names = [n for n in train_features.keys() if n != 'last']
    for lname in layer_names:
        concat_name = f"last+{lname}"
        concat_feat_train = torch.cat([train_features['last'], train_features[lname]], dim=1)
        concat_feat_test = torch.cat([test_features_dict['last'], test_features_dict[lname]], dim=1)
        concat_configs.append((concat_name, concat_feat_train, concat_feat_test))

    # All layers concat
    if len(layer_names) > 0:
        all_feats_train = [train_features['last']]
        all_feats_test = [test_features_dict['last']]
        for lname in layer_names:
            all_feats_train.append(train_features[lname])
            all_feats_test.append(test_features_dict[lname])
        concat_name = "last+" + "+".join(layer_names)
        concat_configs.append((concat_name,
                               torch.cat(all_feats_train, dim=1),
                               torch.cat(all_feats_test, dim=1)))

    for concat_name, concat_train, concat_test in concat_configs:
        D_concat = concat_train.shape[1]

        feat_by_class = defaultdict(list)
        for i in range(concat_train.shape[0]):
            c = train_labels[i].item()
            feat_by_class[c].append(concat_train[i])
        for c in feat_by_class:
            feat_by_class[c] = torch.stack(feat_by_class[c])

        nc = compute_nc_metrics(feat_by_class)
        ncc_acc = compute_ncc_accuracy(feat_by_class, concat_test, test_labels)
        nc['ncc_accuracy'] = round(ncc_acc * 100, 2)

        results[concat_name] = nc

        k_vs_d = "K<D ✅" if D_concat >= 100 else "K>D ❌"
        print(f"\n  [{concat_name}] D={D_concat} → {k_vs_d}")
        print(f"    NC1={nc['nc1']:.4f}  NC2={nc['nc2']:.4f}")
        print(f"    NCC Acc={nc['ncc_accuracy']:.2f}%")

    # --------------------------------------------------------
    # 3 Cases 판별
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print(" 3 Cases Classification")
    print("=" * 80)

    D_last = train_features['last'].shape[1]
    K = 100

    for lname in layer_names:
        D_inter = train_features[lname].shape[1]
        D_concat = D_last + D_inter

        if D_last >= K - 1 and D_inter >= K - 1:
            case = "N/A (both D^L, D^ℓ ≥ K-1, NC already possible)"
        elif D_last < K - 1 and D_inter >= K - 1:
            case = "Case (i): D^L < K-1, D^ℓ ≥ K-1"
        elif D_last < K - 1 and D_inter < K - 1 and D_concat >= K - 1:
            case = "Case (ii): D^L < K-1, D^ℓ < K-1, but D^L+D^ℓ ≥ K-1"
        else:
            case = "Case (iii): D^L+D^ℓ < K-1"

        print(f"  {lname}: D^ℓ={D_inter}, D^L={D_last}, D^L+D^ℓ={D_concat}")
        print(f"  → {case}")

    # --------------------------------------------------------
    # Semantic Hierarchy Analysis
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print(" Semantic Hierarchy: Clustering vs GT Superclass")
    print("=" * 80)

    gt_coarse = [FINE_TO_COARSE[c] for c in range(100)]

    for layer_name, features in train_features.items():
        feat_by_class = defaultdict(list)
        for i in range(features.shape[0]):
            c = train_labels[i].item()
            feat_by_class[c].append(features[i])

        class_means = torch.stack([
            torch.stack(feat_by_class[c]).mean(dim=0) for c in range(100)
        ])

        hier = analyze_hierarchy(class_means, gt_coarse, layer_name)
        results[f"hierarchy_{layer_name}"] = hier

        print(f"  [{layer_name}] NMI={hier['NMI_vs_GT']:.4f}  ARI={hier['ARI_vs_GT']:.4f}")

    # Concat features도 분석
    for concat_name, concat_train, _ in concat_configs:
        feat_by_class = defaultdict(list)
        for i in range(concat_train.shape[0]):
            c = train_labels[i].item()
            feat_by_class[c].append(concat_train[i])

        class_means = torch.stack([
            torch.stack(feat_by_class[c]).mean(dim=0) for c in range(100)
        ])

        hier = analyze_hierarchy(class_means, gt_coarse, concat_name)
        results[f"hierarchy_{concat_name}"] = hier
        print(f"  [{concat_name}] NMI={hier['NMI_vs_GT']:.4f}  ARI={hier['ARI_vs_GT']:.4f}")

    # --------------------------------------------------------
    # Summary Table
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print(" SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Layer':<25} {'D':>5} {'K<D?':>6} {'NC1':>8} {'NC2':>8} {'NC3':>8} {'NCC%':>8} {'NMI':>8}")
    print("-" * 80)

    for key in results:
        if key.startswith('hierarchy_'):
            continue
        r = results[key]
        D = r.get('feature_dim', '')
        kd = '✅' if r.get('K_less_than_D', False) else '❌'
        nc1 = f"{r['nc1']:.3f}" if 'nc1' in r else ''
        nc2 = f"{r['nc2']:.3f}" if 'nc2' in r else ''
        nc3 = f"{r['nc3']:.3f}" if 'nc3' in r else ''
        ncc = f"{r['ncc_accuracy']:.1f}" if 'ncc_accuracy' in r else ''

        hier_key = f"hierarchy_{key}"
        nmi = f"{results[hier_key]['NMI_vs_GT']:.3f}" if hier_key in results else ''

        print(f"{key:<25} {D:>5} {kd:>6} {nc1:>8} {nc2:>8} {nc3:>8} {ncc:>8} {nmi:>8}")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    out_path = 'nc_stats/hnc_validation.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Convert to serializable
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj

    with open(out_path, 'w') as f:
        json.dump(convert(results), f, indent=2)

    print(f"\n>>> Results saved to {out_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()