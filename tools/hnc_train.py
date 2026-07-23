#!/usr/bin/env python
"""
HNC Validation: Universal Training + NC Measurement (v4 — Structure B)
=======================================================================
Structure B: h^ℓ = genuine intermediate layer, h^L = backbone last → FC

4 modes:
  (a) baseline:       h^L only → cls(D^L, K)
  (b) concat:         [h^L; h^ℓ] → cls(D^L+D^ℓ, K)
  (c) random_concat:  [h^L; noise] → cls(D^L+D^ℓ, K)  [control]
  (d) inter_only:     h^ℓ only → cls(D^ℓ, K)

random_concat: same architecture as concat, but h^ℓ replaced with
  gaussian noise each forward pass. If concat > random_concat,
  it proves h^ℓ's NC structure contributes, not just dimension increase.

Model-specific h^ℓ extraction:
  ResNet32:  h^ℓ = GAP(layer2),  h^L = FC(GAP(layer3) → D^L)
  ResNet18:  h^ℓ = GAP(layer3),  h^L = FC(GAP(layer4) → D^L)
  ResNet50:  h^ℓ = GAP(layer3),  h^L = FC(GAP(layer4) → D^L)
  VGG11:     h^ℓ = GAP(block3),  h^L = FC(GAP(block4) → D^L)
  MLP:       h^ℓ = fc2 output,   h^L = fc3 output

Usage:
  python tools/hnc_train.py --dataset cifar100 --case i --mode concat --rank 0
"""
import argparse, json, os, sys, time, logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger()

# ============================================================
# Dataset × Case Configs
# ============================================================
# D_inter = h^ℓ dimension = INTERMEDIATE layer output
# D_last  = h^L dimension = projection FC output
#
# Architecture → intermediate layer:
#   ResNet32:  layer2 = in_ch*2,  in_ch = D_inter // 2
#   ResNet18:  layer3 = 256 (fixed)
#   ResNet50:  layer3 = 1024 (fixed)
#   VGG11:     block3 = ch*2, ch = D_inter // 2
#   MLP:       fc2 = D_inter
#
CONFIGS = {
    # ── CIFAR-100: 3 Cases ──
    ('cifar100', 'i'):   (100, 'resnet32', 128, 64,  200, 128, 0.1),
    ('cifar100', 'ii'):  (100, 'resnet32', 64,  48,  200, 128, 0.1),
    ('cifar100', 'iii'): (100, 'resnet32', 32,  16,  200, 128, 0.1),

    # ── Small datasets ──
    ('mnist', 'i'):      (10,  'mlp',      128, 4,   50,  128, 0.01),
    ('svhn', 'i'):       (10,  'vgg11',    32,  4,   100, 128, 0.05),
    ('cifar10', 'i'):    (10,  'resnet32', 16,  4,   200, 128, 0.1),

    # ── Medium: ResNet18, layer3=256(h^ℓ), layer4=512→FC(h^L) ──
    ('flowers102', 'i'): (102, 'resnet18', 256, 64,  100, 64,  0.01),
    ('food101', 'i'):    (101, 'resnet18', 256, 64,  100, 64,  0.01),

    # ── Large: ResNet50, layer3=1024(h^ℓ), layer4=2048→FC(h^L) ──
    ('cub200', 'i'):     (200, 'resnet50', 1024, 128, 100, 32,  0.01),
    ('tinyimagenet','i'):(200, 'resnet50', 1024, 128, 100, 64,  0.01),
    ('places365', 'i'):  (365, 'resnet50', 1024, 256, 90,  64,  0.01),
    ('imagenet1k', 'i'): (1000,'resnet50', 1024, 512, 90,  256, 0.1),

    # ── Very large ──
    ('inat2018', 'iii'): (8142, 'resnet50', 1024, 1024, 90, 64, 0.01),
    ('inat2021', 'iii'): (10000,'resnet50', 1024, 1024, 90, 64, 0.01),
}


# ============================================================
# Models
# ============================================================
class BasicBlock(nn.Module):
    def __init__(self, in_p, out_p, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_p, out_p, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_p)
        self.conv2 = nn.Conv2d(out_p, out_p, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_p)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_p != out_p:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_p, out_p, 1, stride, bias=False),
                nn.BatchNorm2d(out_p))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


def build_model(arch, K, D_inter, D_last, mode):
    if arch == 'mlp':
        return MLPModel(K, D_inter, D_last, mode)
    elif arch == 'vgg11':
        return VGG11Model(K, D_inter, D_last, mode)
    elif arch == 'resnet32':
        return ResNet32Model(K, D_inter, D_last, mode)
    elif arch in ('resnet18', 'resnet50'):
        return ResNetTVModel(K, D_inter, D_last, mode, arch)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


class MLPModel(nn.Module):
    """
    MLP for MNIST. Structure B:
      fc1(784→256) → fc2(256→D_inter) = h^ℓ → projection(D_inter→D_last) = h^L
    """
    def __init__(self, K, D_inter, D_last, mode):
        super().__init__()
        self.mode = mode
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, D_inter)
        self.projection = nn.Linear(D_inter, D_last)
        self.D_inter = D_inter

        if mode == 'inter_only':
            D_cls = D_inter
        elif mode in ('concat', 'random_concat'):
            D_cls = D_last + D_inter
        else:
            D_cls = D_last
        self.classifier = nn.Linear(D_cls, K)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        h_inter = F.relu(self.fc2(x))
        h_last = self.projection(h_inter)

        if self.mode == 'inter_only':
            logits = self.classifier(h_inter)
        elif self.mode == 'concat':
            logits = self.classifier(torch.cat([h_last, h_inter], 1))
        elif self.mode == 'random_concat':
            noise = torch.randn_like(h_inter)
            logits = self.classifier(torch.cat([h_last, noise], 1))
        else:
            logits = self.classifier(h_last)

        return logits, h_last.detach(), h_inter.detach()


class VGG11Model(nn.Module):
    """
    VGG11 for SVHN. Structure B:
      block1 → block2 → block3 → GAP = h^ℓ
                                  ↓
                         block4 → GAP → FC = h^L
    """
    def __init__(self, K, D_inter, D_last, mode, in_ch=3):
        super().__init__()
        self.mode = mode
        ch = max(D_inter // 2, 8)
        self.D_inter_actual = ch * 2
        self.D_bb = ch * 2

        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1), nn.BatchNorm2d(ch),
            nn.ReLU(), nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch),
            nn.ReLU(), nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(ch, ch*2, 3, padding=1), nn.BatchNorm2d(ch*2), nn.ReLU(),
            nn.Conv2d(ch*2, ch*2, 3, padding=1), nn.BatchNorm2d(ch*2), nn.ReLU(),
            nn.MaxPool2d(2))
        self.block4 = nn.Sequential(
            nn.Conv2d(ch*2, ch*2, 3, padding=1), nn.BatchNorm2d(ch*2), nn.ReLU(),
            nn.Conv2d(ch*2, ch*2, 3, padding=1), nn.BatchNorm2d(ch*2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))
        self.inter_pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(self.D_bb, D_last)

        if mode == 'inter_only':
            D_cls = self.D_inter_actual
        elif mode in ('concat', 'random_concat'):
            D_cls = D_last + self.D_inter_actual
        else:
            D_cls = D_last
        self.classifier = nn.Linear(D_cls, K)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        feat_inter = self.block3(x)
        h_inter = self.inter_pool(feat_inter).flatten(1)
        z_last = self.block4(feat_inter).flatten(1)
        h_last = self.projection(z_last)

        if self.mode == 'inter_only':
            logits = self.classifier(h_inter)
        elif self.mode == 'concat':
            logits = self.classifier(torch.cat([h_last, h_inter], 1))
        elif self.mode == 'random_concat':
            noise = torch.randn_like(h_inter)
            logits = self.classifier(torch.cat([h_last, noise], 1))
        else:
            logits = self.classifier(h_last)

        return logits, h_last.detach(), h_inter.detach()


class ResNet32Model(nn.Module):
    """
    ResNet32 for CIFAR. Structure B:
      conv1 → layer1(in_ch) → layer2(in_ch*2) → GAP = h^ℓ (intermediate)
                                      ↓
                               layer3(in_ch*4) → GAP → FC = h^L (last)

    in_ch = D_inter // 2, so layer2 = D_inter, layer3 = D_inter*2
    """
    def __init__(self, K, D_inter, D_last, mode):
        super().__init__()
        self.mode = mode
        in_ch = D_inter // 2
        assert in_ch * 2 == D_inter, f"D_inter={D_inter} must be divisible by 2"

        D_bb = in_ch * 4
        self.D_bb = D_bb
        self.D_inter = D_inter
        self.D_last = D_last
        self.in_ch = in_ch

        self.conv1 = nn.Conv2d(3, in_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.layer1 = self._make_layer(in_ch, in_ch, 5)
        self.layer2 = self._make_layer(in_ch, in_ch*2, 5, stride=2)
        self.layer3 = self._make_layer(in_ch*2, in_ch*4, 5, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.inter_pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(D_bb, D_last)

        if mode == 'inter_only':
            D_cls = D_inter
        elif mode in ('concat', 'random_concat'):
            D_cls = D_last + D_inter
        else:
            D_cls = D_last
        self.classifier = nn.Linear(D_cls, K)

    def _make_layer(self, inp, outp, n, stride=1):
        return nn.Sequential(BasicBlock(inp, outp, stride),
                             *[BasicBlock(outp, outp) for _ in range(n-1)])

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)

        # h^ℓ = GAP(layer2) — genuine intermediate
        feat_inter = self.layer2(x)
        h_inter = self.inter_pool(feat_inter).flatten(1)

        # h^L = FC(GAP(layer3) → D_last) — non-linear transform after h^ℓ
        feat_last = self.layer3(feat_inter)
        z_last = self.pool(feat_last).flatten(1)
        h_last = self.projection(z_last)

        if self.mode == 'inter_only':
            logits = self.classifier(h_inter)
        elif self.mode == 'concat':
            logits = self.classifier(torch.cat([h_last, h_inter], 1))
        elif self.mode == 'random_concat':
            noise = torch.randn_like(h_inter)
            logits = self.classifier(torch.cat([h_last, noise], 1))
        else:
            logits = self.classifier(h_last)

        return logits, h_last.detach(), h_inter.detach()


class ResNetTVModel(nn.Module):
    """
    ResNet18/50 for medium/large datasets. Structure B:
      stem → layer1 → layer2 → layer3 → GAP = h^ℓ (intermediate)
                                          ↓
                                  layer4 → GAP → FC = h^L (last)

    ResNet18: layer3=256(h^ℓ), layer4=512→FC→D^L(h^L)
    ResNet50: layer3=1024(h^ℓ), layer4=2048→FC→D^L(h^L)
    """
    def __init__(self, K, D_inter, D_last, mode, arch='resnet50'):
        super().__init__()
        self.mode = mode
        base = models.resnet18(weights=None) if arch == 'resnet18' \
            else models.resnet50(weights=None)

        D_inter_natural = 256 if arch == 'resnet18' else 1024
        D_bb = 512 if arch == 'resnet18' else 2048

        assert D_inter == D_inter_natural, \
            f"D_inter={D_inter} must equal {arch} layer3={D_inter_natural}"

        self.D_inter = D_inter
        self.D_bb = D_bb

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1, self.layer2 = base.layer1, base.layer2
        self.layer3, self.layer4 = base.layer3, base.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.inter_pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(D_bb, D_last) \
            if D_bb != D_last else nn.Identity()

        if mode == 'inter_only':
            D_cls = D_inter
        elif mode in ('concat', 'random_concat'):
            D_cls = D_last + D_inter
        else:
            D_cls = D_last
        self.classifier = nn.Linear(D_cls, K)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)

        # h^ℓ = GAP(layer3) — genuine intermediate
        feat_inter = self.layer3(x)
        h_inter = self.inter_pool(feat_inter).flatten(1)

        # h^L = FC(GAP(layer4) → D_last) — non-linear transform after h^ℓ
        feat_last = self.layer4(feat_inter)
        z_last = self.avgpool(feat_last).flatten(1)
        h_last = self.projection(z_last)

        if self.mode == 'inter_only':
            logits = self.classifier(h_inter)
        elif self.mode == 'concat':
            logits = self.classifier(torch.cat([h_last, h_inter], 1))
        elif self.mode == 'random_concat':
            noise = torch.randn_like(h_inter)
            logits = self.classifier(torch.cat([h_last, noise], 1))
        else:
            logits = self.classifier(h_last)

        return logits, h_last.detach(), h_inter.detach()


# ============================================================
# Datasets
# ============================================================
def get_loaders(name, batch, data_root='/data/hoyong'):
    n224 = transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    n32  = transforms.Normalize([.4914,.4822,.4465],[.2023,.1994,.2010])

    if name == 'mnist':
        tf = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.1307,),(0.3081,))])
        return (DataLoader(datasets.MNIST(data_root,True,download=True,transform=tf),
                           batch,True,num_workers=2),
                DataLoader(datasets.MNIST(data_root,False,download=True,transform=tf),
                           batch*2,False,num_workers=2))
    elif name == 'svhn':
        tf = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize([.4377,.4438,.4728],[.198,.201,.197])])
        return (DataLoader(datasets.SVHN(data_root,'train',download=True,transform=tf),
                           batch,True,num_workers=4),
                DataLoader(datasets.SVHN(data_root,'test',download=True,transform=tf),
                           batch*2,False,num_workers=4))
    elif name in ('cifar10','cifar100'):
        cls = datasets.CIFAR10 if name == 'cifar10' else datasets.CIFAR100
        tr = transforms.Compose([transforms.RandomCrop(32,4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(), n32])
        te = transforms.Compose([transforms.ToTensor(), n32])
        return (DataLoader(cls(data_root,True,download=True,transform=tr),
                           batch,True,num_workers=4,pin_memory=True),
                DataLoader(cls(data_root,False,download=True,transform=te),
                           batch*2,False,num_workers=4,pin_memory=True))
    elif name == 'flowers102':
        tr = transforms.Compose([transforms.Resize(256),transforms.RandomCrop(224),
                                 transforms.RandomHorizontalFlip(),transforms.ToTensor(),n224])
        te = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),
                                 transforms.ToTensor(),n224])
        return (DataLoader(datasets.Flowers102(data_root,'train',download=True,transform=tr),
                           batch,True,num_workers=4),
                DataLoader(datasets.Flowers102(data_root,'test',download=True,transform=te),
                           batch*2,False,num_workers=4))
    elif name == 'food101':
        tr = transforms.Compose([transforms.Resize(256),transforms.RandomCrop(224),
                                 transforms.RandomHorizontalFlip(),transforms.ToTensor(),n224])
        te = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),
                                 transforms.ToTensor(),n224])
        return (DataLoader(datasets.Food101(data_root,'train',download=True,transform=tr),
                           batch,True,num_workers=4),
                DataLoader(datasets.Food101(data_root,'test',download=True,transform=te),
                           batch*2,False,num_workers=4))
    elif name == 'tinyimagenet':
        root = os.path.join(data_root, 'tiny-imagenet-200')
        tr = transforms.Compose([transforms.RandomCrop(64,8),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),n224])
        te = transforms.Compose([transforms.ToTensor(),n224])
        return (DataLoader(datasets.ImageFolder(os.path.join(root,'train'),tr),
                           batch,True,num_workers=4,pin_memory=True),
                DataLoader(datasets.ImageFolder(os.path.join(root,'val'),te),
                           batch*2,False,num_workers=4,pin_memory=True))
    else:
        root = os.path.join(data_root, name)
        tr = transforms.Compose([transforms.Resize(256),transforms.RandomCrop(224),
                                 transforms.RandomHorizontalFlip(),transforms.ToTensor(),n224])
        te = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),
                                 transforms.ToTensor(),n224])
        return (DataLoader(datasets.ImageFolder(os.path.join(root,'train'),tr),
                           batch,True,num_workers=4,pin_memory=True),
                DataLoader(datasets.ImageFolder(os.path.join(root,'val'),te),
                           batch*2,False,num_workers=4,pin_memory=True))


def get_train_eval_loader(name, batch, data_root='/data/hoyong'):
    """Train set with TEST-style transforms (no augmentation) for NC measurement."""
    n224 = transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    n32  = transforms.Normalize([.4914,.4822,.4465],[.2023,.1994,.2010])

    if name == 'mnist':
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
        return DataLoader(datasets.MNIST(data_root,True,download=True,transform=tf),
                          batch*2,False,num_workers=2)
    elif name == 'svhn':
        tf = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize([.4377,.4438,.4728],[.198,.201,.197])])
        return DataLoader(datasets.SVHN(data_root,'train',download=True,transform=tf),
                          batch*2,False,num_workers=4)
    elif name in ('cifar10','cifar100'):
        cls = datasets.CIFAR10 if name == 'cifar10' else datasets.CIFAR100
        tf = transforms.Compose([transforms.ToTensor(), n32])
        return DataLoader(cls(data_root,True,download=True,transform=tf),
                          batch*2,False,num_workers=4,pin_memory=True)
    elif name == 'flowers102':
        tf = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),
                                 transforms.ToTensor(),n224])
        return DataLoader(datasets.Flowers102(data_root,'train',download=True,transform=tf),
                          batch*2,False,num_workers=4)
    elif name == 'food101':
        tf = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),
                                 transforms.ToTensor(),n224])
        return DataLoader(datasets.Food101(data_root,'train',download=True,transform=tf),
                          batch*2,False,num_workers=4)
    elif name == 'tinyimagenet':
        root = os.path.join(data_root, 'tiny-imagenet-200')
        tf = transforms.Compose([transforms.ToTensor(),n224])
        return DataLoader(datasets.ImageFolder(os.path.join(root,'train'),tf),
                          batch*2,False,num_workers=4,pin_memory=True)
    else:
        root = os.path.join(data_root, name)
        tf = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),
                                 transforms.ToTensor(),n224])
        return DataLoader(datasets.ImageFolder(os.path.join(root,'train'),tf),
                          batch*2,False,num_workers=4,pin_memory=True)


# ============================================================
# NC Metrics
# ============================================================
def compute_nc(fbc, W=None):
    classes = sorted(fbc.keys())
    K = len(classes)
    if K < 2:
        return {'nc1': float('inf'), 'nc2': float('inf'), 'nc3': 0.0,
                'K': K, 'D': 0, 'K_lt_D': False}
    means = torch.stack([fbc[c].mean(0) for c in classes])
    gm = means.mean(0); D = means.shape[1]; cm = means - gm

    Sw = sum((fbc[c] - means[i]).T @ (fbc[c] - means[i]) / len(fbc[c])
             for i, c in enumerate(classes)) / K
    Sb = cm.T @ cm / K
    try: nc1 = (torch.trace(Sw @ torch.linalg.pinv(Sb)) / K).item()
    except: nc1 = float('inf')

    mn = F.normalize(cm, dim=1); cs = mn @ mn.T
    mask = ~torch.eye(K, dtype=torch.bool)
    nc2 = (cs[mask] - (-1.0 / (K - 1))).abs().mean().item()

    nc3 = None
    if W is not None and W.shape[0] == K and W.shape[1] == D:
        Wc = W - W.mean(0)
        nc3 = round((F.normalize(Wc, dim=1) * F.normalize(cm, dim=1)).sum(1).mean().item(), 4)

    return {'nc1': round(nc1, 4), 'nc2': round(nc2, 4), 'nc3': nc3,
            'K': K, 'D': D, 'K_lt_D': K - 1 < D}


def ncc_acc(fbc, tf, tl):
    classes = sorted(fbc.keys())
    means = torch.stack([fbc[c].mean(0) for c in classes])
    pred = torch.tensor(classes)[
        (F.normalize(tf, dim=1) @ F.normalize(means, dim=1).T).argmax(1)]
    return (pred == tl).float().mean().item()


def semantic_hierarchy(fbc, gt_coarse):
    try:
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        from sklearn.metrics import normalized_mutual_info_score as nmi_fn
        from sklearn.metrics import adjusted_rand_score as ari_fn
    except ImportError:
        return {'NMI': -1.0, 'ARI': -1.0}
    classes = sorted(fbc.keys())
    means = torch.stack([fbc[c].mean(0) for c in classes])
    mn = F.normalize(means - means.mean(0), dim=1).numpy()
    dist = np.maximum(1 - mn @ mn.T, 0); np.fill_diagonal(dist, 0)
    n_groups = len(set(gt_coarse))
    Z = linkage(squareform(dist), method='ward')
    pred = fcluster(Z, t=n_groups, criterion='maxclust')
    gt = [gt_coarse[c] for c in classes]
    return {'NMI': round(nmi_fn(gt, pred), 4), 'ARI': round(ari_fn(gt, pred), 4)}


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--case', default='i', choices=['i', 'ii', 'iii'])
    parser.add_argument('--mode', required=True,
                        choices=['baseline', 'concat', 'random_concat', 'inter_only'])
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--data_root', default='/data/hoyong')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    key = (args.dataset, args.case)
    assert key in CONFIGS, f"No config for {key}. Available: {list(CONFIGS.keys())}"
    K, arch, D_inter, D_last, epochs, batch, lr = CONFIGS[key]
    device = f'cuda:{args.rank}'
    torch.cuda.set_device(args.rank)

    if args.case == 'i':
        assert D_last < K - 1 and D_inter >= K - 1, \
            f"Case (i) violated: D^L={D_last}, D^ℓ={D_inter}, K={K}"
    elif args.case == 'ii':
        assert D_last < K - 1 and D_inter < K - 1 and D_last + D_inter >= K - 1, \
            f"Case (ii) violated: D^L={D_last}, D^ℓ={D_inter}, K={K}"
    elif args.case == 'iii':
        assert D_last < K - 1 and D_inter < K - 1 and D_last + D_inter < K - 1, \
            f"Case (iii) violated: D^L={D_last}, D^ℓ={D_inter}, K={K}"

    exp = f"hnc.{args.dataset}.case{args.case}.{args.mode}.seed{args.seed}"
    out_dir = f'output/hnc_v4/{exp}'
    os.makedirs(out_dir, exist_ok=True)

    log.info(f"\n{'='*70}")
    log.info(f" {exp} [Structure B]")
    log.info(f" K={K}  arch={arch}  D^ℓ(inter)={D_inter}  D^L(last)={D_last}  mode={args.mode}")
    if args.mode in ('concat', 'random_concat'):
        log.info(f" D_cat={D_last+D_inter}, h^ℓ ratio={D_inter/(D_last+D_inter)*100:.0f}%")
    log.info(f"{'='*70}")

    model = build_model(arch, K, D_inter, D_last, args.mode).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"  Params: {n_params:,}")

    # Dimension sanity check
    if arch == 'mlp':
        dummy = torch.randn(2, 1, 28, 28).to(device)
    elif arch in ('resnet32', 'vgg11'):
        dummy = torch.randn(2, 3, 32, 32).to(device)
    else:
        dummy = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        logits, hL, hI = model(dummy)
    log.info(f"  Dim check: logits={logits.shape[1]}, h^L={hL.shape[1]}, h^ℓ={hI.shape[1]}")
    if args.mode in ('concat', 'random_concat'):
        assert hL.shape[1] == D_last, f"h^L dim: {hL.shape[1]} != {D_last}"
        assert hI.shape[1] == D_inter, f"h^ℓ dim: {hI.shape[1]} != {D_inter}"
    del dummy

    train_loader, test_loader = get_loaders(args.dataset, batch, args.data_root)
    train_eval_loader = get_train_eval_loader(args.dataset, batch, args.data_root)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)
    if epochs >= 200:
        milestones = [int(epochs * .5), int(epochs * .75)]
    else:
        milestones = [int(epochs * .6), int(epochs * .8)]
    sched = optim.lr_scheduler.MultiStepLR(optimizer, milestones, 0.1)

    best_acc, best_ep = 0, 0
    for ep in range(1, epochs + 1):
        model.train()
        correct, total = 0, 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            logits, _, _ = model(data)
            F.cross_entropy(logits, target).backward()
            optimizer.step()
            correct += (logits.argmax(1) == target).sum().item()
            total += data.size(0)
        sched.step()

        if ep % max(1, epochs // 20) == 0 or ep == epochs:
            model.eval(); tc = tt = 0
            with torch.no_grad():
                for d, t in test_loader:
                    d, t = d.to(device), t.to(device)
                    tc += (model(d)[0].argmax(1) == t).sum().item()
                    tt += d.size(0)
            acc = tc / tt * 100
            if acc > best_acc:
                best_acc, best_ep = acc, ep
                torch.save(model.state_dict(), f'{out_dir}/best.pth')
            log.info(f"  Ep {ep:3d}/{epochs}  Train {correct/total*100:.1f}%  "
                     f"Test {acc:.2f}%  Best {best_acc:.2f}%")

    # ── NC Measurement ──
    log.info(f"\n  Computing NC metrics...")
    model.load_state_dict(torch.load(f'{out_dir}/best.pth',
                                     map_location=device, weights_only=True))
    model.eval()
    fbc_L, fbc_I, fbc_C = defaultdict(list), defaultdict(list), defaultdict(list)
    tf_L, tf_I, tf_C, tl = [], [], [], []

    with torch.no_grad():
        for data, target in train_eval_loader:
            _, hL, hI = model(data.to(device))
            hL, hI = hL.cpu(), hI.cpu()
            for i in range(data.size(0)):
                c = target[i].item()
                fbc_L[c].append(hL[i])
                fbc_I[c].append(hI[i])
                # Always build concat from REAL features (not noise)
                if args.mode != 'inter_only':
                    fbc_C[c].append(torch.cat([hL[i], hI[i]]))
        for data, target in test_loader:
            _, hL, hI = model(data.to(device))
            hL, hI = hL.cpu(), hI.cpu()
            tf_L.append(hL); tf_I.append(hI)
            if args.mode != 'inter_only':
                tf_C.append(torch.cat([hL, hI], 1))
            tl.append(target)

    for d in [fbc_L, fbc_I, fbc_C]:
        for c in d: d[c] = torch.stack(d[c])
    tf_L, tf_I = torch.cat(tf_L), torch.cat(tf_I)
    if tf_C: tf_C = torch.cat(tf_C)
    tl = torch.cat(tl)

    W = model.classifier.weight.data.cpu()
    R = {'experiment': exp, 'dataset': args.dataset, 'case': args.case,
         'mode': args.mode, 'seed': args.seed, 'version': 'v4_structureB',
         'K': K, 'D_last': D_last, 'D_inter': D_inter,
         'best_acc': round(best_acc, 2), 'best_epoch': best_ep, 'params': n_params}

    # NC measurement: always measure h^L, h^ℓ, and real concat
    if args.mode == 'inter_only':
        measure = [('h_L', fbc_L, tf_L), ('h_l', fbc_I, tf_I)]
    else:
        measure = [('h_L', fbc_L, tf_L), ('h_l', fbc_I, tf_I), ('concat', fbc_C, tf_C)]

    for fn, fb, tf in measure:
        w = None
        if fn == 'h_L' and args.mode == 'baseline': w = W
        if fn == 'h_l' and args.mode == 'inter_only': w = W
        if fn == 'concat' and args.mode == 'concat': w = W
        # random_concat: W is (K × D_cat) but h_L is D_last → dim mismatch
        # NC3 would silently be 0.0 → explicitly None instead
        nc = compute_nc(fb, w)
        nc['ncc_acc'] = round(ncc_acc(fb, tf, tl) * 100, 2)
        R[fn] = nc

    for fn in ['h_L', 'h_l', 'concat']:
        if fn not in R:
            R[fn] = {'nc1': None, 'nc2': None, 'nc3': None,
                     'K': K, 'D': None, 'K_lt_D': None, 'ncc_acc': None}

    # Active feature for gap computation
    af = {'baseline': 'h_L', 'concat': 'concat',
          'random_concat': 'h_L', 'inter_only': 'h_l'}
    active = af[args.mode]
    R['linear_ncc_gap'] = round(best_acc - R[active]['ncc_acc'], 2) \
        if R[active]['ncc_acc'] is not None else None

    # Semantic hierarchy (CIFAR-100)
    C100_F2C = [4,1,14,8,0,6,7,7,18,3,3,14,9,18,7,11,3,9,7,11,6,11,5,10,7,
                6,13,15,3,15,0,11,1,10,12,14,16,9,11,5,5,19,8,8,15,13,14,17,
                18,10,16,4,17,4,2,0,17,4,18,17,10,3,2,12,12,16,12,1,9,19,2,
                10,0,1,16,12,9,13,15,13,16,19,2,4,6,19,5,5,8,19,18,1,2,15,
                6,0,17,8,14,13]
    if args.dataset == 'cifar100':
        for fn, fb, _ in measure:
            if fb: R[f'hier_{fn}'] = semantic_hierarchy(fb, C100_F2C)

    # Print summary
    log.info(f"\n{'─'*70}")
    log.info(f" {exp} | Case {args.case} | Acc {best_acc:.2f}% | Params {n_params:,}")
    log.info(f" {'Feat':<8} {'D':>5} {'K<D':>5} {'NC1':>9} {'NC2':>9} {'NC3':>9} {'NCC%':>8}")
    for fn in ['h_L', 'h_l', 'concat']:
        r = R[fn]
        if r['D'] is None:
            log.info(f" {fn:<8}   (not measured)"); continue
        kd = '✅' if r['K_lt_D'] else '❌'
        nc3_str = f"{r['nc3']:>9.4f}" if r['nc3'] is not None else "      N/A"
        log.info(f" {fn:<8} {r['D']:>5} {kd:>5} {r['nc1']:>9.4f} "
                 f"{r['nc2']:>9.4f} {nc3_str} {r['ncc_acc']:>7.1f}%")
    if R['linear_ncc_gap'] is not None:
        log.info(f" Gap: {R['linear_ncc_gap']:.2f}%")
    if args.dataset == 'cifar100':
        for fn, _, _ in measure:
            h = R.get(f'hier_{fn}', {})
            if h and h.get('NMI', -1) >= 0:
                log.info(f" Hier {fn}: NMI={h['NMI']:.4f} ARI={h['ARI']:.4f}")
    log.info(f"{'─'*70}")
    with open(f'{out_dir}/results.json', 'w') as f: json.dump(R, f, indent=2)
    log.info(f" Saved: {out_dir}/results.json")


if __name__ == '__main__':
    main()