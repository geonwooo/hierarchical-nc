#!/bin/bash
# RunHNCValidation_v3.sh — Complete HNC validation pipeline (v3, bugs fixed)
#
# ★ Changes from v2:
#   - Uses hnc_train.py (all 3 bugs fixed)
#   - Output dir: output/hnc_v3/ (separate from v2 results)
#   - Enlarged: only for ResNet32/VGG11/MLP (not supported for ResNet18/50)
#   - inter_only: included for all datasets
#   - Case (ii)/(iii): all 4 modes including enlarged
#   - Added seed support for reproducibility
#
set -e
cd ~/hierarchical-nc
S="python tools/hnc_train.py"
SEED=0  # Change for multi-seed runs

echo "========================================"
echo " HNC Full Validation Pipeline v3"
echo " Seed: $SEED"
echo " $(date)"
echo "========================================"

# ═══════════════════════════════════════════════════════════════
# Phase 1: Prerequisite (Exp 1 + Exp 2)
# Exp 1: INC in D^L < K-1 → baseline 학습 후 layer별 NC 측정
# Exp 2: D^L sweep → D^L 키우면 NC 복원되는지
# → 이 결과가 논문 전제. 실패하면 STOP.
# ═══════════════════════════════════════════════════════════════
echo ""
echo ">>> Phase 1: CIFAR-100 Case (i) — 4 modes [최우선]"
echo "    baseline: D^L=64, D_bb=128"
echo "    concat:   [64; 128] = 192"
echo "    enlarged: backbone widened, D_bb'=192, in_ch'=48"
echo "    inter_only: D_bb=128"
echo ""
$S --dataset cifar100 --case i --mode baseline   --rank 0 --seed $SEED &
$S --dataset cifar100 --case i --mode concat     --rank 1 --seed $SEED &
$S --dataset cifar100 --case i --mode enlarged   --rank 2 --seed $SEED &
$S --dataset cifar100 --case i --mode inter_only --rank 3 --seed $SEED &
wait
echo ">>> Phase 1 DONE $(date)"
echo ""

# ═══════════════════════════════════════════════════════════════
# Phase 2: CIFAR-100 Case (ii) + Case (iii)
# Case (ii):  D^L=48, D_bb=64, sum=112 ≥ 99
# Case (iii): D^L=16, D_bb=32, sum=48  < 99
# ═══════════════════════════════════════════════════════════════
echo ">>> Phase 2a: CIFAR-100 Case (ii) — 4 modes"
echo "    D^L=48, D_bb=64, concat=112"
$S --dataset cifar100 --case ii --mode baseline   --rank 0 --seed $SEED &
$S --dataset cifar100 --case ii --mode concat     --rank 1 --seed $SEED &
$S --dataset cifar100 --case ii --mode enlarged   --rank 2 --seed $SEED &
$S --dataset cifar100 --case ii --mode inter_only --rank 3 --seed $SEED &
wait
echo ">>> Phase 2a DONE $(date)"
echo ""

echo ">>> Phase 2b: CIFAR-100 Case (iii) — 4 modes"
echo "    D^L=16, D_bb=32, concat=48"
$S --dataset cifar100 --case iii --mode baseline   --rank 0 --seed $SEED &
$S --dataset cifar100 --case iii --mode concat     --rank 1 --seed $SEED &
$S --dataset cifar100 --case iii --mode enlarged   --rank 2 --seed $SEED &
$S --dataset cifar100 --case iii --mode inter_only --rank 3 --seed $SEED &
wait
echo ">>> Phase 2b DONE $(date)"
echo ""

# ═══════════════════════════════════════════════════════════════
# Phase 3: Small datasets (MNIST, CIFAR-10, SVHN)
# All support enlarged (MLP, ResNet32, VGG11)
# ═══════════════════════════════════════════════════════════════
echo ">>> Phase 3a: MNIST (MLP) + CIFAR-10 (ResNet32)"
$S --dataset mnist   --case i --mode baseline   --rank 0 --seed $SEED &
$S --dataset mnist   --case i --mode concat     --rank 1 --seed $SEED &
$S --dataset cifar10 --case i --mode baseline   --rank 2 --seed $SEED &
$S --dataset cifar10 --case i --mode concat     --rank 3 --seed $SEED &
wait
echo ""

echo ">>> Phase 3b: MNIST/CIFAR-10 enlarged + inter_only"
$S --dataset mnist   --case i --mode enlarged   --rank 0 --seed $SEED &
$S --dataset mnist   --case i --mode inter_only --rank 1 --seed $SEED &
$S --dataset cifar10 --case i --mode enlarged   --rank 2 --seed $SEED &
$S --dataset cifar10 --case i --mode inter_only --rank 3 --seed $SEED &
wait
echo ""

echo ">>> Phase 3c: SVHN (VGG11) — 4 modes"
$S --dataset svhn --case i --mode baseline   --rank 0 --seed $SEED &
$S --dataset svhn --case i --mode concat     --rank 1 --seed $SEED &
$S --dataset svhn --case i --mode enlarged   --rank 2 --seed $SEED &
$S --dataset svhn --case i --mode inter_only --rank 3 --seed $SEED &
wait
echo ">>> Phase 3 DONE $(date)"
echo ""

# ═══════════════════════════════════════════════════════════════
# Phase 4: Medium datasets (Flowers, Food) — ResNet18
# NO enlarged (ResNet18 fixed backbone). baseline + concat + inter_only.
# ═══════════════════════════════════════════════════════════════
echo ">>> Phase 4: Flowers-102 + Food-101 (ResNet18)"
echo "    D_bb=512, D^L=64. No enlarged (fixed backbone)."
$S --dataset flowers102 --case i --mode baseline   --rank 0 --seed $SEED &
$S --dataset flowers102 --case i --mode concat     --rank 1 --seed $SEED &
$S --dataset food101    --case i --mode baseline   --rank 2 --seed $SEED &
$S --dataset food101    --case i --mode concat     --rank 3 --seed $SEED &
wait

$S --dataset flowers102 --case i --mode inter_only --rank 0 --seed $SEED &
$S --dataset food101    --case i --mode inter_only --rank 1 --seed $SEED &
wait
echo ">>> Phase 4 DONE $(date)"
echo ""

# ═══════════════════════════════════════════════════════════════
# Phase 5: Medium-large (CUB-200, TinyImageNet) — ResNet50
# NO enlarged. baseline + concat + inter_only.
# ═══════════════════════════════════════════════════════════════
echo ">>> Phase 5: CUB-200 + TinyImageNet (ResNet50)"
echo "    D_bb=2048, D^L=128. No enlarged."
$S --dataset cub200       --case i --mode baseline   --rank 0 --seed $SEED &
$S --dataset cub200       --case i --mode concat     --rank 1 --seed $SEED &
$S --dataset tinyimagenet --case i --mode baseline   --rank 2 --seed $SEED &
$S --dataset tinyimagenet --case i --mode concat     --rank 3 --seed $SEED &
wait

$S --dataset cub200       --case i --mode inter_only --rank 0 --seed $SEED &
$S --dataset tinyimagenet --case i --mode inter_only --rank 1 --seed $SEED &
wait
echo ">>> Phase 5 DONE $(date)"
echo ""

# ═══════════════════════════════════════════════════════════════
# Phase 6: Large (Places365, ImageNet-1K) — ResNet50
# ═══════════════════════════════════════════════════════════════
echo ">>> Phase 6: Places365 + ImageNet-1K (ResNet50)"
echo "    D_bb=2048, D^L=256/512. No enlarged."
$S --dataset places365  --case i --mode baseline --rank 0 --seed $SEED &
$S --dataset places365  --case i --mode concat   --rank 1 --seed $SEED &
$S --dataset imagenet1k --case i --mode baseline --rank 2 --seed $SEED &
$S --dataset imagenet1k --case i --mode concat   --rank 3 --seed $SEED &
wait

$S --dataset places365  --case i --mode inter_only --rank 0 --seed $SEED &
$S --dataset imagenet1k --case i --mode inter_only --rank 1 --seed $SEED &
wait
echo ">>> Phase 6 DONE $(date)"
echo ""

# ═══════════════════════════════════════════════════════════════
# Phase 7: iNaturalist (Case iii, natural D constraint)
# ═══════════════════════════════════════════════════════════════
echo ">>> Phase 7: iNaturalist (Case iii)"
echo "    D_bb=2048, D^L=1024. D^L+D_bb=3072 < K."
$S --dataset inat2018 --case iii --mode baseline --rank 0 --seed $SEED &
$S --dataset inat2018 --case iii --mode concat   --rank 1 --seed $SEED &
$S --dataset inat2021 --case iii --mode baseline --rank 2 --seed $SEED &
$S --dataset inat2021 --case iii --mode concat   --rank 3 --seed $SEED &
wait

$S --dataset inat2018 --case iii --mode inter_only --rank 0 --seed $SEED &
$S --dataset inat2021 --case iii --mode inter_only --rank 1 --seed $SEED &
wait
echo ">>> Phase 7 DONE $(date)"
echo ""

# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
echo ""
echo "========================================"
echo " ALL RESULTS SUMMARY (v3)"
echo "========================================"
echo ""
echo "Dataset         Case  Mode          Acc%    NC1_L     NC1_l    NC1_cat   NC3      NCC%     Gap%"
echo "──────────────────────────────────────────────────────────────────────────────────────────────────"

for f in output/hnc_v3/hnc.*/results.json; do
    [ -f "$f" ] || continue
    python3 -c "
import json
with open('$f') as fp: r=json.load(fp)
m=r['mode']; d=r['dataset']; c=r['case']
hL=r.get('h_L',{}); hl=r.get('h_l',{}); cat=r.get('concat',{})
nc1L = f\"{hL.get('nc1',''):>8}\" if hL.get('nc1') is not None else '     N/A'
nc1l = f\"{hl.get('nc1',''):>8}\" if hl.get('nc1') is not None else '     N/A'
nc1c = f\"{cat.get('nc1',''):>8}\" if cat.get('nc1') is not None else '     N/A'
nc3  = f\"{cat.get('nc3', hL.get('nc3','')):>8}\" if (cat.get('nc3') or hL.get('nc3')) is not None else '     N/A'
ncc  = f\"{cat.get('ncc_acc', hL.get('ncc_acc','')):>7}\" if (cat.get('ncc_acc') or hL.get('ncc_acc')) is not None else '    N/A'
gap  = f\"{r.get('linear_ncc_gap',''):>7}\" if r.get('linear_ncc_gap') is not None else '    N/A'
print(f\"{d:<15} {c:<5} {m:<13} {r['best_acc']:>6.2f}  {nc1L}  {nc1l}  {nc1c}  {nc3}  {ncc}%  {gap}%\")
" 2>/dev/null
done

echo ""
echo "========================================"
echo " COMPLETE $(date)"
echo " Version: v3 (all bugs fixed)"
echo "========================================"