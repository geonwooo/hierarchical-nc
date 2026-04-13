"""
Nearest Class Mean evaluation + per-level NC stats.

Usage:
  python tools/ncm_eval.py \
      --cfg configs/cifar100/seq_base.yaml \
      --checkpoint output/cifar100/.../best_model.pth \
      --output nc_stats/result.json --rank 0
"""
import argparse, json, os, sys
import numpy as np, torch
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'main'))
import _init_paths
from config import cfg, update_config
from utils.utils import get_model
import dataset as custom_dataset
from data_transform.transform_wrapper import get_transform
from torch.utils.data import DataLoader

def compute_nc_metrics(class_means, class_features, K, D, classifier_weight=None):
    global_mean = class_means.mean(axis=0)
    Sw = np.zeros((D, D)); count = 0
    for c in range(K):
        if c not in class_features or len(class_features[c]) == 0: continue
        feats = np.array(class_features[c])
        centered = feats - class_means[c]
        Sw += centered.T @ centered; count += len(feats)
    Sw /= max(count, 1)
    centered_means = class_means - global_mean
    Sb = centered_means.T @ centered_means / K
    try: nc1 = np.trace(Sw @ np.linalg.pinv(Sb)) / K
    except: nc1 = float('inf')
    normed = centered_means / np.maximum(np.linalg.norm(centered_means, axis=1, keepdims=True), 1e-8)
    cos_sim = normed @ normed.T
    mask = ~np.eye(K, dtype=bool)
    nc2 = np.mean(np.abs(cos_sim[mask] - (-1.0/(K-1))))
    nc3 = None
    if classifier_weight is not None:
        W = classifier_weight if classifier_weight.shape[0] == K else classifier_weight.T
        if W.shape[0] == K:
            nc3 = float(np.linalg.norm(W/(np.linalg.norm(W,'fro')+1e-8) - centered_means/(np.linalg.norm(centered_means,'fro')+1e-8), 'fro'))
    return {'nc1': float(nc1), 'nc2': float(nc2), 'nc3': nc3, 'K': K, 'D': D, 'K_lt_D': K < D}

def ncm_accuracy(class_means, features_list, targets_list):
    normed_means = class_means / np.maximum(np.linalg.norm(class_means, axis=1, keepdims=True), 1e-8)
    correct, total = 0, 0
    for feats, tgts in zip(features_list, targets_list):
        feats_np = np.array(feats)
        normed = feats_np / np.maximum(np.linalg.norm(feats_np, axis=1, keepdims=True), 1e-8)
        preds = np.argmax(normed @ normed_means.T, axis=1)
        correct += (preds == np.array(tgts)).sum(); total += len(tgts)
    return correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--grouping', type=str, default=None)
    args_cli = parser.parse_args()
    class FakeArgs:
        def __init__(self, c, g=None):
            self.cfg = c; self.opts = ['dataset.grouping_file', g] if g else []
    update_config(cfg, FakeArgs(args_cli.cfg, args_cli.grouping))
    rank = args_cli.rank; torch.cuda.set_device(rank)
    num_classes = cfg.dataset.num_classes
    model = get_model(cfg, num_classes, rank)
    ckpt = torch.load(args_cli.checkpoint, map_location='cuda:{}'.format(rank))
    state = {k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(state); model.eval()
    mm = model.module if hasattr(model, 'module') else model
    transform = get_transform(cfg, mode='test')
    train_set = custom_dataset.CIFAR100(cfg, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=256, shuffle=False, num_workers=2)
    test_set = custom_dataset.CIFAR100(cfg, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)
    print("Extracting features...")
    class_features = defaultdict(list)
    with torch.no_grad():
        for batch in train_loader:
            data, targets = batch[0].cuda(rank), batch[1]
            features = mm.extract_feature(data)
            for i in range(features.size(0)):
                class_features[targets[i].item()].append(features[i].cpu().numpy())
    D = len(class_features[0][0])
    class_means = np.zeros((num_classes, D))
    for c in range(num_classes):
        if len(class_features[c]) > 0: class_means[c] = np.mean(class_features[c], axis=0)
    test_features, test_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            data, targets = batch[0].cuda(rank), batch[1]
            features = mm.extract_feature(data)
            test_features.append(features.cpu().numpy()); test_targets.append(targets.numpy())
    W_fine = mm.classifier.weight.detach().cpu().numpy() if hasattr(mm, 'classifier') else None
    fine_nc = compute_nc_metrics(class_means, class_features, num_classes, D, W_fine)
    fine_nc['ncm_acc'] = float(ncm_accuracy(class_means, test_features, test_targets))
    results = {'fine': fine_nc}
    for k, v in fine_nc.items(): print("  {}: {}".format(k, v))
    if cfg.dataset.num_classes_1 > 0:
        from builder.network import Network
        f2c = Network._get_fine_to_coarse(cfg); num_groups = len(set(f2c))
        group_features = defaultdict(list)
        for c in range(num_classes): group_features[f2c[c]].extend(class_features[c])
        group_means = np.zeros((num_groups, D))
        for g in range(num_groups):
            if len(group_features[g]) > 0: group_means[g] = np.mean(group_features[g], axis=0)
        W_c = mm.classifier_coarse.weight.detach().cpu().numpy() if hasattr(mm, 'classifier_coarse') else None
        coarse_nc = compute_nc_metrics(group_means, group_features, num_groups, D, W_c)
        coarse_nc['ncm_acc'] = float(ncm_accuracy(group_means, test_features, [np.array([f2c[t] for t in tgts]) for tgts in test_targets]))
        results['coarse'] = coarse_nc
    linear_acc = ckpt.get('best_result', None)
    if linear_acc:
        results['linear_ncm_gap'] = float(linear_acc - results['fine']['ncm_acc'])
        print("  Linear: {:.2f}% NCM: {:.2f}% Gap: {:.2f}%".format(linear_acc*100, results['fine']['ncm_acc']*100, results['linear_ncm_gap']*100))
    os.makedirs(os.path.dirname(args_cli.output), exist_ok=True)
    with open(args_cli.output, 'w') as f: json.dump(results, f, indent=2)
    print("Saved: {}".format(args_cli.output))

if __name__ == '__main__': main()
