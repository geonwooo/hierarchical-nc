#!/usr/bin/env python
"""
HNC Validation: Universal Training + NC Measurement (v3 — all bugs fixed)
==========================================================================
Fixes from v2:
  1. [CONCAT]   h^ℓ = backbone LAST layer GAP output (projection 전)
                 h^L = FC(h^ℓ → D^L) (projection 후)
                 → 같은 feature의 압축 전/후를 concat (v2는 layer2를 썼음)
  2. [ENLARGED] backbone 자체를 넓힘 (v2는 bottleneck만 넓혔음)
                 in_ch' = (D^L + D_bb) // 4, no projection
  3. [IN_CH]    ResNet32: in_ch = D_inter // 4 (v2는 // 2)
                 → layer3 output = in_ch*4 = D_inter = D_bb
  4. [CONFIGS]  ResNet18: D_inter=512, ResNet50: D_inter=2048 (실제 backbone output)
                 Case (ii)/(iii): D_last 수정 (projection이 identity가 되는 문제)

4 modes × 12+ datasets, 3 Cases for CIFAR-100

Usage:
  python tools/hnc_train_v3.py --dataset cifar100 --case i --mode concat --rank 0
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
# Key: (dataset, case)
# Value: (K, arch, D_inter, D_last, epochs, batch, lr)
#
# ★ CRITICAL CHANGE from v2:
#   D_inter = D_bb = backbone LAST layer output dim (= D^ℓ)
#   D_last  = D^L  = projection FC output dim
#
# v2 bugs:
#   - D_inter was layer2 output for ResNet32 (should be layer3)
#   - D_inter was layer3 output for ResNetTV (should be layer4)
#   - Case (ii)/(iii): D_last == D_inter → projection was Identity
#
# Architecture → D_bb mapping:
#   ResNet32:  layer3 = in_ch * 4,  in_ch = D_inter // 4
#   ResNet18:  layer4 = 512 (fixed)
#   ResNet50:  layer4 = 2048 (fixed)
#   VGG11:     block4 = ch * 2,     ch = D_inter // 2
#   MLP:       fc2 output = D_inter (direct)
#
CONFIGS = {
    # ── CIFAR-100: 3 Cases ──
    # Case (i):  D^ℓ=128 ≥ K-1=99,  D^L=64 < 99
    ('cifar100', 'i'):   (100, 'resnet32', 128, 64,  200, 128, 0.1),
    #   in_ch=32, layer1=32, layer2=64, layer3=128=D_bb
    #   FC(128→64)=projection, classifier(64→100) or concat(192→100)

    # Case (ii): D^ℓ=64 < 99, D^L=48 < 99, D^L+D^ℓ=112 ≥ 99
    ('cifar100', 'ii'):  (100, 'resnet32', 64,  48,  200, 128, 0.1),
    #   in_ch=16, layer3=64=D_bb, FC(64→48)=projection  ★ v2: D_last=64 (bug)

    # Case (iii): D^ℓ=32 < 99, D^L=16 < 99, D^L+D^ℓ=48 < 99
    ('cifar100', 'iii'): (100, 'resnet32', 32,  16,  200, 128, 0.1),
    #   in_ch=8,  layer3=32=D_bb, FC(32→16)=projection  ★ v2: D_last=32 (bug)

    # ── Small datasets: all Case (i) ──
    ('mnist', 'i'):      (10,  'mlp',      128, 4,   50,  128, 0.01),
    #   MLP: fc1(784→256), fc2(256→128=D_bb), FC(128→4)=projection

    ('svhn', 'i'):       (10,  'vgg11',    32,  4,   100, 128, 0.05),
    #   VGG11: ch=16, block4=32=D_bb, FC(32→4)=projection

    ('cifar10', 'i'):    (10,  'resnet32', 16,  4,   200, 128, 0.1),
    #   in_ch=4, layer3=16=D_bb, FC(16→4)=projection

    # ── Medium datasets: ResNet18, D_bb=512 (fixed backbone) ──
    ('flowers102', 'i'): (102, 'resnet18', 512, 64,  100, 64,  0.01),
    #   ★ v2: D_inter=256 (layer3, bug). Now 512 (layer4, correct)

    ('food101', 'i'):    (101, 'resnet18', 512, 64,  100, 64,  0.01),

    # ── Medium-large: ResNet50, D_bb=2048 (fixed backbone) ──
    ('cub200', 'i'):     (200, 'resnet50', 2048, 128, 100, 32,  0.01),
    #   ★ v2: D_inter=1024 (layer3, bug). Now 2048 (layer4, correct)

    ('tinyimagenet','i'):(200, 'resnet50', 2048, 128, 100, 64,  0.01),

    # ── Large datasets ──
    ('places365', 'i'):  (365, 'resnet50', 2048, 256, 90,  64,  0.01),

    ('imagenet1k', 'i'): (1000,'resnet50', 2048, 512, 90,  256, 0.1),

    # ── Very large: Case (iii) naturally ──
    # D^ℓ=2048 < K-1, D^L=1024 < K-1, sum=3072 < K-1
    ('inat2018', 'iii'): (8142, 'resnet50', 2048, 1024, 90, 64, 0.01),
    #   ★ v2: D_inter=1024, D_last=2048 (swapped, bug)

    ('inat2021', 'iii'): (10000,'resnet50', 2048, 1024, 90, 64, 0.01),
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
    """
    Build model for given mode.

    D_inter = D_bb = backbone LAST layer output dim (= D^ℓ)
    D_last  = D^L  = projection FC output dim

    ★ v2 bug: build_model modified D_last_eff for enlarged, but didn't
      change backbone width → bottleneck widening only, not backbone widening.
    ★ v3 fix: mode is passed directly to each Model class, which handles
      backbone widening internally for enlarged mode.
    """
    if arch == 'mlp':
        return MLPModel(K, D_inter, D_last, mode)
    elif arch == 'vgg11':
        return VGG11Model(K, D_inter, D_last, mode)
    elif arch == 'resnet32':
        return ResNet32Model(K, D_inter, D_last, mode)
    elif arch in ('resnet18', 'resnet50'):
        if mode == 'enlarged':
            raise NotImplementedError(
                f"Enlarged not implemented for {arch} (fixed backbone width). "
                f"Enlarged is supported for resnet32, vgg11, mlp only.")
        return ResNetTVModel(K, D_inter, D_last, mode, arch)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


class MLPModel(nn.Module):
    """
    MLP for MNIST.

    Structure:
      fc1(784→256) → fc2(256→D_bb) → [projection FC(D_bb→D^L)] → classifier
      h^ℓ = fc2 output (= D_bb, projection 전)
      h^L = projection output (= D^L, projection 후)

    ★ This was already correct in v2 — fc2 output IS the last hidden
      layer before projection. No change needed in extraction logic.
    """
    def __init__(self, K, D_inter, D_last, mode):
        super().__init__()
        self.mode = mode

        if mode == 'enlarged':
            # ★ v3: wider hidden layer, no projection
            D_bb = D_last + D_inter
            self.fc1 = nn.Linear(784, 256)
            self.fc2 = nn.Linear(256, D_bb)
            self.projection = None
            D_cls = D_bb
        else:
            D_bb = D_inter
            self.fc1 = nn.Linear(784, 256)
            self.fc2 = nn.Linear(256, D_bb)
            self.projection = nn.Linear(D_bb, D_last) if D_bb != D_last else nn.Identity()
            if mode == 'inter_only':
                D_cls = D_bb
            elif mode == 'concat':
                D_cls = D_last + D_bb
            else:  # baseline
                D_cls = D_last

        self.D_bb = D_bb
        self.classifier = nn.Linear(D_cls, K)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        z = F.relu(self.fc2(x))  # backbone last output

        if self.mode == 'enlarged':
            logits = self.classifier(z)
            return logits, z.detach(), z.detach()

        h_inter = z                    # h^ℓ = projection 전
        h_last = self.projection(z)    # h^L = projection 후

        if self.mode == 'inter_only':
            logits = self.classifier(h_inter)
        elif self.mode == 'concat':
            logits = self.classifier(torch.cat([h_last, h_inter], 1))
        else:
            logits = self.classifier(h_last)

        return logits, h_last.detach(), h_inter.detach()


class VGG11Model(nn.Module):
    """
    VGG11-like for SVHN.

    ★ v2 bug:  h^ℓ = pool(block2 output), a MIDDLE layer
               → block3/4가 이미 흡수한 정보 → concat 무의미
    ★ v3 fix:  h^ℓ = block4 GAP output (backbone LAST output, projection 전)
               h^L = FC(h^ℓ → D^L) (projection 후)
               ch = D_inter // 2 (was max(D_inter, 8))
               → block4 output = ch*2 = D_inter = D_bb
    """
    def __init__(self, K, D_inter, D_last, mode, in_ch=3):
        super().__init__()
        self.mode = mode

        if mode == 'enlarged':
            D_bb = D_last + D_inter
        else:
            D_bb = D_inter

        ch = max(D_bb // 2, 8)
        actual_D_bb = ch * 2
        self.D_bb = actual_D_bb

        if actual_D_bb != D_bb:
            log.warning(f"VGG11: requested D_bb={D_bb}, actual={actual_D_bb} (ch={ch})")

        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(), nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(), nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(ch, ch*2, 3, padding=1), nn.BatchNorm2d(ch*2), nn.ReLU(),
            nn.Conv2d(ch*2, ch*2, 3, padding=1), nn.BatchNorm2d(ch*2), nn.ReLU(), nn.MaxPool2d(2))
        self.block4 = nn.Sequential(
            nn.Conv2d(ch*2, ch*2, 3, padding=1), nn.BatchNorm2d(ch*2), nn.ReLU(),
            nn.Conv2d(ch*2, ch*2, 3, padding=1), nn.BatchNorm2d(ch*2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))

        if mode == 'enlarged':
            self.projection = None
            D_cls = actual_D_bb
        else:
            self.projection = nn.Linear(actual_D_bb, D_last)
            if mode == 'inter_only':
                D_cls = actual_D_bb
            elif mode == 'concat':
                D_cls = D_last + actual_D_bb
            else:
                D_cls = D_last

        self.classifier = nn.Linear(D_cls, K)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # ★ v3 fix: h^ℓ = block4 GAP output (backbone LAST, projection 전)
        # ★ v2 bug: h^ℓ = pool(block2 output) (MIDDLE layer)
        z = self.block4(x).flatten(1)  # D_bb

        if self.mode == 'enlarged':
            logits = self.classifier(z)
            return logits, z.detach(), z.detach()

        h_inter = z                     # h^ℓ = projection 전
        h_last = self.projection(z)     # h^L = projection 후

        if self.mode == 'inter_only':
            logits = self.classifier(h_inter)
        elif self.mode == 'concat':
            logits = self.classifier(torch.cat([h_last, h_inter], 1))
        else:
            logits = self.classifier(h_last)

        return logits, h_last.detach(), h_inter.detach()


class ResNet32Model(nn.Module):
    """
    ResNet32 for CIFAR.

    ★ v2 bugs (ALL THREE):
      1. in_ch = D_inter // 2 → layer2 = D_inter, layer3 = 2*D_inter
         h^ℓ was layer2 output, h^L was FC(layer3→D_last)
         → h^ℓ와 h^L이 서로 다른 stage → layer3가 h^ℓ 정보 흡수 → concat 무의미
      2. Enlarged: same backbone + FC(D_raw→192) = bottleneck 완화
         → backbone 안 넓어짐 → 공정한 control 아님
      3. D_inter_actual 계산 오류

    ★ v3 fix:
      1. in_ch = D_inter // 4 → layer3 = D_inter = D_bb
         h^ℓ = GAP(layer3) = D_bb (projection 전, 같은 feature)
         h^L = FC(D_bb → D_last) (projection 후)
      2. Enlarged: in_ch' = (D_last + D_inter) // 4
         → backbone 자체가 넓어짐, bottleneck 없음
      3. D_bb tracked correctly

    Dimensions:
      conv1:  3 → in_ch
      layer1: in_ch → in_ch     (5 blocks)
      layer2: in_ch → in_ch*2   (5 blocks, stride 2)
      layer3: in_ch*2 → in_ch*4 (5 blocks, stride 2)
      GAP → in_ch*4 = D_bb

      baseline/concat/inter_only: in_ch = D_inter // 4,  D_bb = D_inter
      enlarged:                   in_ch = (D_last+D_inter) // 4,  D_bb = D_last+D_inter
    """
    def __init__(self, K, D_inter, D_last, mode):
        super().__init__()
        self.mode = mode

        if mode == 'enlarged':
            # ★ v3: backbone 자체를 넓힘
            D_bb = D_last + D_inter
            in_ch = D_bb // 4
            assert in_ch * 4 == D_bb, \
                f"(D_last+D_inter)={D_bb} must be divisible by 4, got {D_bb}"
        else:
            # ★ v3: in_ch = D_inter // 4 (v2 was // 2)
            D_bb = D_inter
            in_ch = D_inter // 4
            assert in_ch * 4 == D_inter, \
                f"D_inter={D_inter} must be divisible by 4, got {D_inter}"

        self.D_bb = D_bb
        self.D_last = D_last
        self.in_ch = in_ch

        self.conv1 = nn.Conv2d(3, in_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.layer1 = self._make_layer(in_ch, in_ch, 5)
        self.layer2 = self._make_layer(in_ch, in_ch*2, 5, stride=2)
        self.layer3 = self._make_layer(in_ch*2, in_ch*4, 5, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)

        if mode == 'enlarged':
            # ★ v3: no projection, backbone output directly to classifier
            self.projection = None
            D_cls = D_bb
        else:
            self.projection = nn.Linear(D_bb, D_last) if D_bb != D_last else nn.Identity()
            if mode == 'inter_only':
                D_cls = D_bb
            elif mode == 'concat':
                D_cls = D_last + D_bb
            else:  # baseline
                D_cls = D_last

        self.classifier = nn.Linear(D_cls, K)

    def _make_layer(self, inp, outp, n, stride=1):
        return nn.Sequential(BasicBlock(inp, outp, stride),
                             *[BasicBlock(outp, outp) for _ in range(n-1)])

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # ★ v3 fix: h^ℓ = GAP(layer3) = backbone LAST output
        # ★ v2 bug: h^ℓ was GAP(layer2), a MIDDLE layer
        z = self.pool(x).flatten(1)  # D_bb = in_ch*4

        if self.mode == 'enlarged':
            logits = self.classifier(z)
            return logits, z.detach(), z.detach()

        h_inter = z                     # h^ℓ = D_bb (projection 전)
        h_last = self.projection(z)     # h^L = D_last (projection 후)

        if self.mode == 'inter_only':
            logits = self.classifier(h_inter)
        elif self.mode == 'concat':
            logits = self.classifier(torch.cat([h_last, h_inter], 1))
        else:  # baseline
            logits = self.classifier(h_last)

        return logits, h_last.detach(), h_inter.detach()


class ResNetTVModel(nn.Module):
    """
    ResNet18/50 using torchvision (for non-CIFAR datasets).

    ★ v2 bug:  h^ℓ = pool(layer3 output), a MIDDLE layer
               D_inter was set to layer3 dim (256 for R18, 1024 for R50)
    ★ v3 fix:  h^ℓ = GAP(layer4 output) = backbone LAST output
               D_inter must match actual backbone output:
                 ResNet18: D_inter = 512
                 ResNet50: D_inter = 2048

    Enlarged: NOT SUPPORTED for torchvision models (fixed backbone width).
              Use ResNet32 for enlarged experiments.
    """
    def __init__(self, K, D_inter, D_last, mode, arch='resnet50'):
        super().__init__()
        self.mode = mode
        base = models.resnet18(weights=None) if arch == 'resnet18' \
            else models.resnet50(weights=None)

        D_bb_natural = 512 if arch == 'resnet18' else 2048
        assert D_inter == D_bb_natural, \
            f"D_inter={D_inter} must equal {arch} layer4 output={D_bb_natural}. " \
            f"★ v2 had D_inter=layer3 output, v3 uses layer4."

        self.D_bb = D_bb_natural

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1, self.layer2 = base.layer1, base.layer2
        self.layer3, self.layer4 = base.layer3, base.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.projection = nn.Linear(D_bb_natural, D_last) \
            if D_bb_natural != D_last else nn.Identity()

        if mode == 'inter_only':
            D_cls = D_bb_natural
        elif mode == 'concat':
            D_cls = D_last + D_bb_natural
        else:  # baseline
            D_cls = D_last

        self.classifier = nn.Linear(D_cls, K)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # ★ v3 fix: h^ℓ = GAP(layer4) = backbone LAST output
        # ★ v2 bug: h^ℓ was pool(layer3), a MIDDLE layer
        x = self.layer4(x)
        z = self.avgpool(x).flatten(1)  # D_bb (512 or 2048)

        h_inter = z                     # h^ℓ = projection 전
        h_last = self.projection(z)     # h^L = projection 후

        if self.mode == 'inter_only':
            logits = self.classifier(h_inter)
        elif self.mode == 'concat':
            logits = self.classifier(torch.cat([h_last, h_inter], 1))
        else:  # baseline
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


# ============================================================
# NC Metrics
# ============================================================
def compute_nc(fbc, W=None):
    """Compute NC1, NC2, NC3 from per-class feature banks."""
    classes = sorted(fbc.keys())
    K = len(classes)
    if K < 2:
        return {'nc1': float('inf'), 'nc2': float('inf'), 'nc3': 0.0,
                'K': K, 'D': 0, 'K_lt_D': False}

    means = torch.stack([fbc[c].mean(0) for c in classes])
    gm = means.mean(0)
    D = means.shape[1]
    cm = means - gm

    # NC1: within-class variability collapse
    Sw = sum((fbc[c] - means[i]).T @ (fbc[c] - means[i]) / len(fbc[c])
             for i, c in enumerate(classes)) / K
    Sb = cm.T @ cm / K
    try:
        nc1 = (torch.trace(Sw @ torch.linalg.pinv(Sb)) / K).item()
    except Exception:
        nc1 = float('inf')

    # NC2: simplex ETF
    mn = F.normalize(cm, dim=1)
    cs = mn @ mn.T
    mask = ~torch.eye(K, dtype=torch.bool)
    nc2 = (cs[mask] - (-1.0 / (K - 1))).abs().mean().item()

    # NC3: self-duality (only when classifier weight matches feature dim)
    nc3 = 0.0
    if W is not None and W.shape[0] == K and W.shape[1] == D:
        Wc = W - W.mean(0)
        nc3 = (F.normalize(Wc, dim=1) * F.normalize(cm, dim=1)).sum(1).mean().item()

    return {'nc1': round(nc1, 4), 'nc2': round(nc2, 4), 'nc3': round(nc3, 4),
            'K': K, 'D': D, 'K_lt_D': K - 1 < D}


def ncc_acc(fbc, tf, tl):
    """Nearest Class Center accuracy."""
    classes = sorted(fbc.keys())
    means = torch.stack([fbc[c].mean(0) for c in classes])
    pred = torch.tensor(classes)[
        (F.normalize(tf, dim=1) @ F.normalize(means, dim=1).T).argmax(1)]
    return (pred == tl).float().mean().item()


def semantic_hierarchy(fbc, gt_coarse):
    """Measure clustering alignment with ground-truth hierarchy."""
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
    dist = np.maximum(1 - mn @ mn.T, 0)
    np.fill_diagonal(dist, 0)
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
                        choices=['baseline', 'concat', 'enlarged', 'inter_only'])
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--data_root', default='/data/hoyong')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # Seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    key = (args.dataset, args.case)
    assert key in CONFIGS, f"No config for {key}. Available: {list(CONFIGS.keys())}"
    K, arch, D_inter, D_last, epochs, batch, lr = CONFIGS[key]
    device = f'cuda:{args.rank}'
    torch.cuda.set_device(args.rank)

    # Verify case conditions
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
    out_dir = f'output/hnc_v3/{exp}'
    os.makedirs(out_dir, exist_ok=True)

    log.info(f"\n{'='*70}")
    log.info(f" {exp}")
    log.info(f" K={K}  arch={arch}  D^ℓ(D_bb)={D_inter}  D^L={D_last}  mode={args.mode}")
    if args.mode == 'enlarged':
        D_enlarged = D_last + D_inter
        log.info(f" Enlarged: D_bb'={D_enlarged}, backbone widened")
    elif args.mode == 'concat':
        log.info(f" Concat: D_cat={D_last+D_inter}")
    log.info(f"{'='*70}")

    model = build_model(arch, K, D_inter, D_last, args.mode).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"  Params: {n_params:,}")

    # ── Dimension sanity check ──
    dummy = torch.randn(2, 1, 28, 28).to(device) if arch == 'mlp' \
        else torch.randn(2, 3, 32, 32).to(device) if arch in ('resnet32', 'vgg11') \
        else torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        logits, hL, hI = model(dummy)
    log.info(f"  Dim check: logits={logits.shape[1]}, h^L={hL.shape[1]}, h^ℓ={hI.shape[1]}")
    if args.mode == 'concat':
        assert hL.shape[1] == D_last, f"h^L dim mismatch: {hL.shape[1]} != {D_last}"
        assert hI.shape[1] == D_inter, f"h^ℓ dim mismatch: {hI.shape[1]} != {D_inter}"
        assert logits.shape[1] == K
    elif args.mode == 'baseline':
        assert hL.shape[1] == D_last, f"h^L dim mismatch: {hL.shape[1]} != {D_last}"
    elif args.mode == 'enlarged':
        assert hL.shape[1] == D_last + D_inter, \
            f"enlarged dim mismatch: {hL.shape[1]} != {D_last+D_inter}"
    del dummy

    # ── Data ──
    train_loader, test_loader = get_loaders(args.dataset, batch, args.data_root)

    # ── Optimizer & Scheduler ──
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)
    if epochs >= 200:
        milestones = [int(epochs * .5), int(epochs * .75)]
    else:
        milestones = [int(epochs * .6), int(epochs * .8)]
    sched = optim.lr_scheduler.MultiStepLR(optimizer, milestones, 0.1)

    # ── Training ──
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
            model.eval()
            tc = tt = 0
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

    fbc_L = defaultdict(list)   # per-class h^L features
    fbc_I = defaultdict(list)   # per-class h^ℓ features
    fbc_C = defaultdict(list)   # per-class [h^L; h^ℓ] features
    tf_L, tf_I, tf_C, tl = [], [], [], []

    with torch.no_grad():
        for data, target in train_loader:
            _, hL, hI = model(data.to(device))
            hL, hI = hL.cpu(), hI.cpu()
            for i in range(data.size(0)):
                c = target[i].item()
                fbc_L[c].append(hL[i])
                fbc_I[c].append(hI[i])
                if args.mode not in ('enlarged', 'inter_only'):
                    fbc_C[c].append(torch.cat([hL[i], hI[i]]))

        for data, target in test_loader:
            _, hL, hI = model(data.to(device))
            hL, hI = hL.cpu(), hI.cpu()
            tf_L.append(hL)
            tf_I.append(hI)
            if args.mode not in ('enlarged', 'inter_only'):
                tf_C.append(torch.cat([hL, hI], 1))
            tl.append(target)

    for d in [fbc_L, fbc_I, fbc_C]:
        for c in d:
            d[c] = torch.stack(d[c])
    tf_L = torch.cat(tf_L)
    tf_I = torch.cat(tf_I)
    if tf_C:
        tf_C = torch.cat(tf_C)
    tl = torch.cat(tl)

    # ── NC computation ──
    W = model.classifier.weight.data.cpu()

    R = {
        'experiment': exp, 'dataset': args.dataset, 'case': args.case,
        'mode': args.mode, 'seed': args.seed,
        'K': K, 'D_last': D_last, 'D_inter': D_inter,
        'best_acc': round(best_acc, 2), 'best_epoch': best_ep,
        'params': n_params,
        'version': 'v3',
    }

    # Which feature spaces to measure NC on (skip meaningless ones)
    if args.mode == 'enlarged':
        # Only h_L is meaningful (= enlarged backbone output)
        measure = [('h_L', fbc_L, tf_L)]
    elif args.mode == 'inter_only':
        # h_L and h_l are same dim but measure both for completeness
        measure = [('h_L', fbc_L, tf_L), ('h_l', fbc_I, tf_I)]
    else:
        # baseline/concat: measure all 3 spaces
        measure = [('h_L', fbc_L, tf_L), ('h_l', fbc_I, tf_I), ('concat', fbc_C, tf_C)]

    for fn, fb, tf in measure:
        # Determine if classifier weight matches this feature space for NC3
        w = None
        if fn == 'h_L' and args.mode in ('baseline', 'enlarged'):
            w = W
        if fn == 'h_l' and args.mode == 'inter_only':
            w = W
        if fn == 'concat' and args.mode == 'concat':
            w = W

        nc = compute_nc(fb, w)
        nc['ncc_acc'] = round(ncc_acc(fb, tf, tl) * 100, 2)
        R[fn] = nc

    # Fill missing spaces with empty dict for consistent output
    for fn in ['h_L', 'h_l', 'concat']:
        if fn not in R:
            R[fn] = {'nc1': None, 'nc2': None, 'nc3': None,
                     'K': K, 'D': None, 'K_lt_D': None, 'ncc_acc': None}

    # Active feature for gap computation
    af = {'baseline': 'h_L', 'concat': 'concat', 'enlarged': 'h_L', 'inter_only': 'h_l'}
    active = af[args.mode]
    if R[active]['ncc_acc'] is not None:
        R['linear_ncc_gap'] = round(best_acc - R[active]['ncc_acc'], 2)
    else:
        R['linear_ncc_gap'] = None

    # ── Semantic hierarchy (CIFAR-100 only) ──
    C100_F2C = [4,1,14,8,0,6,7,7,18,3,3,14,9,18,7,11,3,9,7,11,6,11,5,10,7,
                6,13,15,3,15,0,11,1,10,12,14,16,9,11,5,5,19,8,8,15,13,14,17,
                18,10,16,4,17,4,2,0,17,4,18,17,10,3,2,12,12,16,12,1,9,19,2,
                10,0,1,16,12,9,13,15,13,16,19,2,4,6,19,5,5,8,19,18,1,2,15,
                6,0,17,8,14,13]
    if args.dataset == 'cifar100':
        for fn, fb, _ in measure:
            if fb:
                R[f'hier_{fn}'] = semantic_hierarchy(fb, C100_F2C)

    # ── Print summary ──
    log.info(f"\n{'─'*70}")
    log.info(f" {exp} | Case {args.case} | Acc {best_acc:.2f}% | Params {n_params:,}")
    log.info(f" {'Feat':<8} {'D':>5} {'K<D':>5} {'NC1':>9} {'NC2':>9} "
             f"{'NC3':>9} {'NCC%':>8}")
    for fn in ['h_L', 'h_l', 'concat']:
        r = R[fn]
        if r['D'] is None:
            log.info(f" {fn:<8}   (not measured)")
            continue
        kd = '✅' if r['K_lt_D'] else '❌'
        log.info(f" {fn:<8} {r['D']:>5} {kd:>5} {r['nc1']:>9.4f} "
                 f"{r['nc2']:>9.4f} {r['nc3']:>9.4f} {r['ncc_acc']:>7.1f}%")
    if R['linear_ncc_gap'] is not None:
        log.info(f" Gap: {R['linear_ncc_gap']:.2f}%")

    if args.dataset == 'cifar100':
        for fn, _, _ in measure:
            h = R.get(f'hier_{fn}', {})
            if h and h.get('NMI', -1) >= 0:
                log.info(f" Hier {fn}: NMI={h['NMI']:.4f} ARI={h['ARI']:.4f}")

    log.info(f"{'─'*70}")
    with open(f'{out_dir}/results.json', 'w') as f:
        json.dump(R, f, indent=2)
    log.info(f" Saved: {out_dir}/results.json")


if __name__ == '__main__':
    main()