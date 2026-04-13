#!/bin/bash
# RunHNCValidation.sh — 12 datasets × 2 modes = 24 experiments
# 4 GPU parallel, auto-scheduling
set -e
cd ~/hierarchical-nc

SCRIPT="python tools/hnc_train.py"

echo "============================================"
echo " HNC Validation: 12 Datasets × 2 Modes"
echo " 4 GPU parallel"
echo "============================================"

# ── Phase 1: Small datasets (MNIST, CIFAR-10, CIFAR-100, SVHN) ──
echo ">>> Phase 1: Small datasets (~2 hours)"

$SCRIPT --dataset mnist    --mode baseline --rank 0 &
$SCRIPT --dataset mnist    --mode concat   --rank 1 &
$SCRIPT --dataset cifar10  --mode baseline --rank 2 &
$SCRIPT --dataset cifar10  --mode concat   --rank 3 &
wait; echo ">>> MNIST + CIFAR-10 done"

$SCRIPT --dataset cifar100 --mode baseline --rank 0 &
$SCRIPT --dataset cifar100 --mode concat   --rank 1 &
$SCRIPT --dataset svhn     --mode baseline --rank 2 &
$SCRIPT --dataset svhn     --mode concat   --rank 3 &
wait; echo ">>> CIFAR-100 + SVHN done"

# ── Phase 2: Medium datasets (Flowers, Food, TinyImageNet, CUB) ──
echo ">>> Phase 2: Medium datasets (~6 hours)"

$SCRIPT --dataset flowers102   --mode baseline --rank 0 &
$SCRIPT --dataset flowers102   --mode concat   --rank 1 &
$SCRIPT --dataset food101      --mode baseline --rank 2 &
$SCRIPT --dataset food101      --mode concat   --rank 3 &
wait; echo ">>> Flowers + Food done"

$SCRIPT --dataset tinyimagenet --mode baseline --rank 0 &
$SCRIPT --dataset tinyimagenet --mode concat   --rank 1 &
$SCRIPT --dataset cub200       --mode baseline --rank 2 &
$SCRIPT --dataset cub200       --mode concat   --rank 3 &
wait; echo ">>> TinyImageNet + CUB done"

# ── Phase 3: Large datasets (Places365, ImageNet-1K) ──
echo ">>> Phase 3: Large datasets (~48 hours)"

$SCRIPT --dataset places365  --mode baseline --rank 0 &
$SCRIPT --dataset places365  --mode concat   --rank 1 &
$SCRIPT --dataset imagenet1k --mode baseline --rank 2 &
$SCRIPT --dataset imagenet1k --mode concat   --rank 3 &
wait; echo ">>> Places365 + ImageNet done"

# ── Phase 4: Very large datasets (iNat2018, iNat2021) ──
echo ">>> Phase 4: iNaturalist (~48 hours)"

$SCRIPT --dataset inat2018 --mode baseline --rank 0 &
$SCRIPT --dataset inat2018 --mode concat   --rank 1 &
$SCRIPT --dataset inat2021 --mode baseline --rank 2 &
$SCRIPT --dataset inat2021 --mode concat   --rank 3 &
wait; echo ">>> iNaturalist done"

# ── Results Summary ──
echo ""
echo "============================================"
echo " ALL RESULTS"
echo "============================================"
printf "%-20s %-10s %6s %6s %8s %8s %8s %8s %8s\n" \
    "Dataset" "Mode" "Acc%" "Case" "NC1_L" "NC1_I" "NC1_C" "NC3" "NCC%"
echo "────────────────────────────────────────────────────────────────────────────"

for f in output/hnc/hnc.*/nc_results.json; do
    [ -f "$f" ] || continue
    python -c "
import json
with open('$f') as fp:
    r = json.load(fp)
print(f\"{r['dataset']:<20} {r['mode']:<10} {r['best_acc']:>6.1f} {r['case']:<6} {r['last']['nc1']:>8.3f} {r['inter']['nc1']:>8.3f} {r['concat']['nc1']:>8.3f} {r.get('concat',{}).get('nc3',0):>8.3f} {r['concat']['ncc_acc']:>7.1f}%\")
" 2>/dev/null
done

echo "============================================"
